# RESIDE-6K Outdoor fixed split

- `train/{hazy,clear}`: 2718 pairs.
- `val/{hazy,clear}`: 282 pairs.
- `test/{hazy,clear}`: 500 pairs.

Train and validation were split from the supplied 3000-pair outdoor training partition by clear-scene ID using seed 2026 and a 10% validation target. `split_manifest.csv` records every selected pair. The test directory is preserved from the supplied RESIDE-6K Outdoor test set; it is the associated SOTS Outdoor version and should be reported as one test set.
