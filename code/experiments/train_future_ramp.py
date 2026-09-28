from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


CLASSES = ["non_ramp", "ramp_up", "ramp_down"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


class RampDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path):
        self.frame=frame.reset_index(drop=True); self.cache_dir=cache_dir; self.arrays={}
    def __len__(self): return len(self.frame)
    def _array(self, shard):
        if shard not in self.arrays: self.arrays[shard]=np.load(self.cache_dir/f'{shard}_pixels_128.npy',mmap_mode='r')
        return self.arrays[shard]
    def __getitem__(self,index):
        r=self.frame.iloc[index]; a=self._array(str(r.shard)); image=np.asarray(a[int(r.shard_row)],dtype=np.float32).copy()
        image=torch.from_numpy(image).permute(2,0,1).div_(255.0)
        return image, torch.tensor(CLASS_TO_ID[str(r.ramp_label)],dtype=torch.long), int(r._row_id)


class RampClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__(); w=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone=efficientnet_b0(weights=w); dim=self.backbone.classifier[1].in_features; self.backbone.classifier=nn.Identity()
        self.head=nn.Sequential(nn.Linear(dim,256),nn.SiLU(),nn.Dropout(.2),nn.Linear(256,3))
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1)); self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1))
    def forward(self,x): return self.head(self.backbone((x-self.mean)/self.std))


def classification_metrics(y: np.ndarray, p: np.ndarray) -> tuple[pd.DataFrame,dict]:
    rows=[]
    for i,name in enumerate(CLASSES):
        tp=int(((y==i)&(p==i)).sum()); fp=int(((y!=i)&(p==i)).sum()); fn=int(((y==i)&(p!=i)).sum()); support=int((y==i).sum())
        precision=tp/(tp+fp) if tp+fp else 0.; recall=tp/(tp+fn) if tp+fn else 0.; f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        rows.append({'class':name,'support':support,'precision':precision,'recall':recall,'f1':f1})
    per=pd.DataFrame(rows); overall={'n':len(y),'accuracy':float((y==p).mean()),'balanced_accuracy':float(per.recall.mean()),'macro_f1':float(per.f1.mean())}
    return per,overall


@torch.no_grad()
def predict(model,loader,device):
    model.eval(); ids=[]; ys=[]; probs=[]
    for images,labels,rowids in loader:
        logits=model(images.to(device,non_blocking=True)); ids.append(rowids.numpy()); ys.append(labels.numpy()); probs.append(torch.softmax(logits,1).cpu().numpy())
    return np.concatenate(ids),np.concatenate(ys),np.concatenate(probs)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--features',required=True); p.add_argument('--cache-dir',required=True); p.add_argument('--ramp-labels',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--patience',type=int,default=4); p.add_argument('--batch-size',type=int,default=160); p.add_argument('--workers',type=int,default=8); p.add_argument('--seed',type=int,default=20260822); a=p.parse_args()
    seed_all(a.seed); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); started=time.time()
    features_path=Path(a.features); labels_path=Path(a.ramp_labels); cache=Path(a.cache_dir)
    f=pd.read_csv(features_path,low_memory=False)
    l=pd.read_csv(labels_path,usecols=['sample_id','start_time','end_time','eligible','ramp_label'])
    x=f.merge(l,on='sample_id',how='inner',validate='one_to_one'); x['_row_id']=np.arange(len(x)); x['end_time']=pd.to_datetime(x.end_time); x['image_timestamp']=pd.to_datetime(x.image_timestamp)
    x=x.loc[x.eligible.astype(bool)&x.ramp_label.isin(CLASSES)].copy()
    # Forward target must terminate inside its own chronological partition.
    split_end=x.groupby('split_chronological').image_timestamp.max().to_dict()
    x['split_end']=x.split_chronological.map(split_end)
    x=x.loc[x.end_time.le(x.split_end)].copy()
    if x.sample_id.duplicated().any() or x.ramp_label.isna().any(): raise ValueError('Invalid labels after purge')
    parts={name:x.loc[x.split_chronological.eq(name)].copy() for name in ['train','validation','test']}
    if any(len(z)==0 for z in parts.values()): raise ValueError('Empty partition')
    train_count=parts['train'].ramp_label.value_counts().reindex(CLASSES)
    weights=np.sqrt(len(parts['train'])/(len(CLASSES)*train_count.to_numpy(float))); weights=weights/weights.mean()
    datasets={k:RampDataset(v,cache) for k,v in parts.items()}
    loaders={k:DataLoader(v,batch_size=a.batch_size,shuffle=k=='train',num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0,drop_last=False,generator=torch.Generator().manual_seed(a.seed)) for k,v in datasets.items()}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    if device.type!='cuda': raise RuntimeError('GPU required')
    model=RampClassifier(True).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=1e-6); criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device),label_smoothing=.02); scaler=torch.amp.GradScaler('cuda')
    history=[]; best=-1.; best_epoch=0; stale=0; checkpoint=out/'best_model.pt'
    for epoch in range(1,a.epochs+1):
        model.train(); total=0.; n=0
        for images,labels,_ in loaders['train']:
            images=images.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'): loss=criterion(model(images),labels)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),5.); scaler.step(opt); scaler.update(); total+=float(loss.detach())*len(labels); n+=len(labels)
        ids,y,prob=predict(model,loaders['validation'],device); pred=prob.argmax(1); _,m=classification_metrics(y,pred)
        row={'epoch':epoch,'train_loss':total/n,'learning_rate':opt.param_groups[0]['lr'],**{f'val_{k}':v for k,v in m.items()}}; history.append(row); print(json.dumps(row),flush=True)
        score=m['macro_f1']
        if score>best+1e-5:
            best=score; best_epoch=epoch; stale=0; torch.save({'state_dict':model.state_dict(),'epoch':epoch,'validation_macro_f1':score,'args':vars(a),'class_weights':weights.tolist()},checkpoint)
        else: stale+=1
        sched.step()
        if stale>=a.patience: break
    ck=torch.load(checkpoint,map_location=device,weights_only=False); model.load_state_dict(ck['state_dict']); history=pd.DataFrame(history); history.to_csv(out/'training_history.csv',index=False)
    all_predictions=[]; metrics_rows=[]; per_rows=[]
    for split,loader in loaders.items():
        ids,y,prob=predict(model,loader,device); pred=prob.argmax(1); per,m=classification_metrics(y,pred); m['split']=split; metrics_rows.append(m); per['split']=split; per_rows.append(per)
        z=x.set_index('_row_id').loc[ids].reset_index(); z['observed_class_id']=y; z['predicted_class_id']=pred; z['observed_label']=[CLASSES[i] for i in y]; z['predicted_label']=[CLASSES[i] for i in pred]
        for i,name in enumerate(CLASSES): z[f'prob_{name}']=prob[:,i]
        z['split']=split; all_predictions.append(z)
    predictions=pd.concat(all_predictions,ignore_index=True); predictions.to_csv(out/'predictions.csv',index=False); metric=pd.DataFrame(metrics_rows); metric.to_csv(out/'metrics.csv',index=False); per=pd.concat(per_rows,ignore_index=True); per.to_csv(out/'per_class_metrics.csv',index=False)
    test=predictions.loc[predictions.split.eq('test')]; cm=pd.crosstab(pd.Categorical(test.observed_label,categories=CLASSES),pd.Categorical(test.predicted_label,categories=CLASSES),dropna=False); cm.to_csv(out/'test_confusion_matrix.csv')
    test_prob=test[[f'prob_{c}' for c in CLASSES]].to_numpy(); test_y=test.observed_class_id.to_numpy(); onehot=np.eye(3)[test_y]; brier=float(np.mean(np.sum((test_prob-onehot)**2,axis=1)))
    majority=np.zeros_like(test_y); _,major=classification_metrics(test_y,majority)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2)); axes[0].plot(history.epoch,history.val_macro_f1,marker='o'); axes[0].axvline(best_epoch,color='gray',ls='--'); axes[0].set(xlabel='Epoch',ylabel='Validation Macro-F1'); axes[0].grid(alpha=.25); axes[1].imshow(cm.to_numpy(),cmap='Blues'); axes[1].set_xticks(range(3),CLASSES,rotation=25); axes[1].set_yticks(range(3),CLASSES); axes[1].set(xlabel='Predicted',ylabel='Observed');
    for i in range(3):
        for j in range(3): axes[1].text(j,i,str(cm.iloc[i,j]),ha='center',va='center',color='white' if cm.iloc[i,j]>cm.to_numpy().max()/2 else 'black')
    fig.tight_layout(); fig.savefig(out/'training_and_confusion.png',dpi=180); plt.close(fig)
    summary={'schema_version':'1.0-future-ramp-single-image','forecast_horizon_minutes':15,'threshold_kw':1.5,'inputs':'single_current_sky_image_only','seed':a.seed,'best_epoch':best_epoch,'validation_macro_f1':best,'runtime_minutes':(time.time()-started)/60,'class_weights_train_only':dict(zip(CLASSES,weights.tolist())),'partition_counts':{k:int(len(v)) for k,v in parts.items()},'metrics':metrics_rows,'test_multiclass_brier':brier,'majority_baseline_test':major,'input_sha256':{'features':sha256(features_path),'ramp_labels':sha256(labels_path)}}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    hashes=[]
    for q in sorted(z for z in out.iterdir() if z.is_file() and z.name!='SHA256SUMS.txt'): hashes.append(f'{sha256(q)}  {q.name}')
    (out/'SHA256SUMS.txt').write_text('\n'.join(hashes)+'\n')


if __name__=='__main__': main()
