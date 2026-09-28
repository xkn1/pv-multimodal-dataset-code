# Known annotation limitations

The final release contains 500 image/mask pairs with mask values 0-3.

A consistency heuristic flagged six records:

- CLOUD_0344
- CLOUD_0111
- CLOUD_0315
- CLOUD_0114
- CLOUD_0117
- CLOUD_0241

Visual review confirmed that these masks classify nearly the full valid sky region as cloud even though blue gaps are visible. The records are retained for transparency and are listed in `data/annotations/consistency_flags.csv`. Users should exclude these six records from semantic-segmentation benchmarking unless the masks are independently re-annotated. They may still be used for tasks that do not rely on the mask as ground truth.
