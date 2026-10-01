#!/usr/bin/env python3
"""Attach directional cloud-change evidence to 15-min PV ramps and plot examples."""
from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


CAPACITY_KW = 15.0
POWER_THRESHOLD_KW = 1.5


def raw_crop(raw: bytes, size: int = 512) -> Image.Image:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    cx, cy, r90 = 2000.0, 1484.0, 1409.0
    radius = r90 * math.tan(math.radians(73.0 / 2.0))
    box = tuple(int(round(v)) for v in (cx-radius, cy-radius, cx+radius, cy+radius))
    crop = image.crop(box).resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(crop).copy()
    yy, xx = np.mgrid[:size, :size]; center=(size-1)/2; valid=(xx-center)**2+(yy-center)**2 <= (size*.492)**2
    arr[~valid] = 0
    return Image.fromarray(arr)


def classify_support(row, cloud_thr, sun_thr):
    direction = row.ramp_label
    if direction not in ("ramp_up", "ramp_down") or not np.isfinite(row.delta_cloud_fraction):
        return "not_applicable", 0.0, False, False
    sign = 1.0 if direction == "ramp_down" else -1.0
    cloud_evidence = sign * row.delta_cloud_fraction
    sun_evidence = sign * (-row.delta_sun_peak_probability)
    cloud_hit = cloud_evidence >= cloud_thr
    sun_hit = sun_evidence >= sun_thr
    reverse = (cloud_evidence <= -cloud_thr) or (sun_evidence <= -sun_thr)
    score = cloud_evidence / cloud_thr + sun_evidence / sun_thr
    if cloud_hit and sun_hit: label = "strong_cloud_supported"
    elif cloud_hit or sun_hit: label = "cloud_supported"
    elif reverse: label = "contradictory"
    else: label = "ambiguous"
    return label, float(score), bool(cloud_hit), bool(sun_hit)


def choose_examples(events, per_direction=2):
    chosen=[]; used_days=set()
    for direction in ("ramp_down", "ramp_up"):
        pool=events[(events.ramp_label==direction)&events.cloud_power_label.isin(["strong_cloud_supported","cloud_supported"])].copy()
        pool=pool.sort_values(["selection_score","abs_delta_kw"],ascending=False)
        count=0
        for _,row in pool.iterrows():
            day=str(row.start_time)[:10]
            if day in used_days: continue
            chosen.append(row); used_days.add(day); count+=1
            if count>=per_direction: break
    return pd.DataFrame(chosen)


def plot_event(row, timeline, archives, output):
    start=pd.Timestamp(row.start_time); end=pd.Timestamp(row.end_time)
    lo=start-pd.Timedelta(minutes=45); hi=end+pd.Timedelta(minutes=45)
    window=timeline[(timeline.power_timestamp>=lo)&(timeline.power_timestamp<=hi)].copy()
    candidates=window[(window.power_timestamp>=start-pd.Timedelta(minutes=15))&(window.power_timestamp<=end+pd.Timedelta(minutes=15))]
    desired=pd.date_range(start-pd.Timedelta(minutes=15),end+pd.Timedelta(minutes=15),periods=5)
    indices=[]
    for t in desired:
        idx=(candidates.power_timestamp-t).abs().idxmin()
        if idx not in indices: indices.append(idx)
    selected=timeline.loc[indices].sort_values("power_timestamp")

    # Reserve a separate band between the time ticks and the two-line image captions.
    # tight_layout() can undo this spacing when the secondary y-axis is present.
    fig=plt.figure(figsize=(13.2,8.0))
    grid=fig.add_gridspec(2,len(selected),height_ratios=[2.15,1],
                          left=.075,right=.925,top=.94,bottom=.065,hspace=.38,wspace=.04)
    ax=fig.add_subplot(grid[0,:]); ax_cloud=ax.twinx()
    ax.plot(window.power_timestamp,window.power_operational_qc_kw,"o-",ms=3,lw=2,color="#1f77b4",label="PV power")
    ax_cloud.plot(window.power_timestamp,window.cloud_fraction_prob,"-",lw=1.8,color="#7f3c8d",label="Cloud fraction")
    ax_cloud.plot(window.power_timestamp,window.sun_peak_probability,"--",lw=1.4,color="#e6ab02",label="Sun visibility probability")
    shade="#d62728" if row.ramp_label=="ramp_down" else "#2ca02c"
    ax.axvspan(start,end,color=shade,alpha=.10,label="15-min ramp window")
    ax.axvline(start,color=shade,ls="--",lw=1); ax.axvline(end,color=shade,ls="--",lw=1)
    title=(f"{row.ramp_label}: ΔP={row.delta_kw:+.2f} kW, Δcloud={row.delta_cloud_fraction:+.3f}, "
           f"Δsun={row.delta_sun_peak_probability:+.3f} — {row.cloud_power_label}")
    ax.set_title(title); ax.set_ylabel("PV power (kW)"); ax_cloud.set_ylabel("Probability / fraction")
    ax_cloud.set_ylim(-.03,1.03); ax.grid(alpha=.22)
    # The timezone is part of the figure-level label, below both panels, so it
    # never collides with the centre thumbnail's timestamp and power caption.
    fig.text(.5,.025,"Local time (Asia/Shanghai)",ha="center",va="center",fontsize=10)
    handles,labels=ax.get_legend_handles_labels(); h2,l2=ax_cloud.get_legend_handles_labels()
    ax.legend(handles+h2,labels+l2,loc="upper left",ncol=2,fontsize=9)
    image_axes=[]
    for col,(_,sample) in enumerate(selected.iterrows()):
        iax=fig.add_subplot(grid[1,col]); image_axes.append(iax)
        container=sample.image_container
        if container not in archives: archives[container]=zipfile.ZipFile(container)
        image=raw_crop(archives[container].read(sample.image_member))
        iax.imshow(image); iax.axis("off")
        iax.set_title(pd.Timestamp(sample.power_timestamp).strftime("%H:%M")+f"\nP={sample.power_operational_qc_kw:.1f} kW, C={sample.cloud_fraction_prob:.2f}",fontsize=9)
        ax.axvline(sample.power_timestamp,color="0.55",ls=":",lw=.9)
    fig.savefig(output,dpi=190,bbox_inches="tight"); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--features",required=True); p.add_argument("--cloud-features",required=True)
    p.add_argument("--ramp-labels",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--per-direction",type=int,default=2)
    a=p.parse_args(); out=Path(a.output_dir); figures=out/"figures"; figures.mkdir(parents=True,exist_ok=True)
    features=pd.read_csv(a.features,low_memory=False).reset_index(drop=True)
    clouds=pd.read_csv(a.cloud_features,low_memory=False).sort_values("row_index").reset_index(drop=True)
    ramps=pd.read_csv(a.ramp_labels,low_memory=False)
    if len(features)!=len(clouds) or features.sample_id.tolist()!=clouds.sample_id.tolist(): raise RuntimeError("Feature/cloud alignment failed")
    cloud_cols=["sample_id","cloud_fraction_prob","sun_peak_probability","sun_visible_fraction_p05","cloud_entropy","ensemble_disagreement"]
    frame=features.merge(clouds[cloud_cols],on="sample_id",validate="one_to_one")
    frame["power_timestamp"]=pd.to_datetime(frame.image_timestamp).dt.round("5min")
    labels=ramps.merge(clouds[cloud_cols],on="sample_id",validate="one_to_one")
    labels["start_time"]=pd.to_datetime(labels.start_time); labels["end_time"]=pd.to_datetime(labels.end_time)
    end_map=frame[["power_timestamp","cloud_fraction_prob","sun_peak_probability","sun_visible_fraction_p05"]].drop_duplicates("power_timestamp")
    end_map=end_map.rename(columns={"power_timestamp":"end_time","cloud_fraction_prob":"end_cloud_fraction_prob",
        "sun_peak_probability":"end_sun_peak_probability","sun_visible_fraction_p05":"end_sun_visible_fraction_p05"})
    labels=labels.merge(end_map,on="end_time",how="left",validate="many_to_one")
    labels["delta_cloud_fraction"]=labels.end_cloud_fraction_prob-labels.cloud_fraction_prob
    labels["delta_sun_peak_probability"]=labels.end_sun_peak_probability-labels.sun_peak_probability
    labels["delta_sun_visible_fraction"]=labels.end_sun_visible_fraction_p05-labels.sun_visible_fraction_p05
    train=labels[(labels.split_chronological=="train")&labels.ramp_eligible.astype(bool)&labels.delta_cloud_fraction.notna()]
    cloud_thr=max(.10,float(train.delta_cloud_fraction.abs().quantile(.70)))
    sun_thr=max(.15,float(train.delta_sun_peak_probability.abs().quantile(.70)))
    classified=labels.apply(lambda r: classify_support(r,cloud_thr,sun_thr),axis=1,result_type="expand")
    classified.columns=["cloud_power_label","cloud_direction_score","cloud_threshold_hit","sun_threshold_hit"]
    labels=pd.concat([labels,classified],axis=1); labels["abs_delta_kw"]=labels.delta_kw.abs()
    labels["selection_score"]=labels.abs_delta_kw/POWER_THRESHOLD_KW + labels.cloud_direction_score.clip(lower=0) - labels.ensemble_disagreement
    labels.to_csv(out/"cloud_power_ramp_labels_all.csv",index=False)

    candidates=labels[(labels.split_chronological=="test")&labels.ramp_event.astype(bool)&(labels.solar_elevation_deg>=20)&
                      labels.end_cloud_fraction_prob.notna()].copy()
    selected=choose_examples(candidates,a.per_direction)
    if len(selected)<2*a.per_direction: raise RuntimeError(f"Only selected {len(selected)} examples")
    selected.to_csv(out/"selected_typical_events.csv",index=False)
    archives={}
    try:
        for rank,(_,row) in enumerate(selected.iterrows(),1):
            name=f"{rank:02d}_{row.ramp_label}_{str(row.start_time)[:10]}.png"
            plot_event(row,frame,archives,figures/name)
    finally:
        for archive in archives.values(): archive.close()

    event_counts=(labels[labels.ramp_event.astype(bool)].groupby(["ramp_label","cloud_power_label"]).size().rename("count").reset_index())
    event_counts.to_csv(out/"cloud_power_window_counts.csv",index=False)
    thresholds={"power_horizon_minutes":15,"power_threshold_kw":POWER_THRESHOLD_KW,"power_threshold_capacity_fraction":.10,
        "cloud_change_abs_threshold_train_q70_with_floor":cloud_thr,"sun_peak_change_abs_threshold_train_q70_with_floor":sun_thr,
        "cloud_threshold_floor":.10,"sun_threshold_floor":.15,"selection_split":"test","minimum_solar_elevation_deg":20}
    (out/"thresholds.json").write_text(json.dumps(thresholds,ensure_ascii=False,indent=2),encoding="utf-8")
    report=f"""# 云量—功率联合斜坡标签与典型样本报告

## 标签原则

主斜坡标签保持预注册定义：15 min 净功率变化达到 ±1.5 kW。云量不参与修改功率事件真值，只用于增加云致证据标签，避免循环定义。本表为逐预报起点的15 min候选窗口标签，相邻窗口允许重叠；它不等同于已合并去重的6,405个事件。

- `strong_cloud_supported`：云量变化和太阳可见性变化都与功率方向一致且超过训练期阈值；
- `cloud_supported`：两类证据至少一类超过阈值；
- `ambiguous`：有功率爬坡，但云证据较弱；
- `contradictory`：主要云证据与功率变化方向相反，可能涉及太阳高度趋势、设备状态、分割误差或其他因素。

云量阈值为 {cloud_thr:.4f}，太阳峰值概率阈值为 {sun_thr:.4f}；均由 chronological train 的绝对15 min变化第70百分位并设置物理下限得到，测试集不参与定阈值。

## 典型图筛选

仅从时间隔离测试集选择，要求太阳高度≥20°、主斜坡标签成立、云端点完整。排序综合功率变化幅度、方向一致的云证据和模型集成不确定性；同一日期最多选一个事件。最终选择 {len(selected)} 个事件，上升/下降各 {a.per_direction} 个。

## 解释边界

这些标签表示“云证据与功率爬坡方向一致”，不能单凭相关时间变化证明严格因果关系。论文中建议使用 `cloud-associated ramp` 或 `cloud-supported ramp`，不要直接写成“云导致”的绝对真值。
"""
    (out/"云量功率联合斜坡事件报告.md").write_text(report,encoding="utf-8")
    print(json.dumps({"thresholds":thresholds,"selected":selected[["sample_id","start_time","end_time","ramp_label","delta_kw","delta_cloud_fraction","delta_sun_peak_probability","cloud_power_label","selection_score"]].to_dict("records"),"counts":event_counts.to_dict("records")},ensure_ascii=False,default=str),flush=True)


if __name__=="__main__": main()
