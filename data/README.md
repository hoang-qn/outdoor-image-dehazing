# Curated outdoor dehazing data

- `RESIDE-6K-Outdoor/{train,val,test}/{hazy,clear}`: 2718 / 282 / 500 paired images.
- `O-HAZE/{hazy,clear}`: 45 real outdoor haze pairs; external test set.
- `NH-HAZE/{hazy,clear}`: 55 real outdoor non-homogeneous haze pairs; external test set.
- `HD-NH-HAZE/Train/{hazy,clean}`: 40 real outdoor pairs at 6000x4000. Its `pairs.csv` assigns `ntire23-001`–`005` to validation and `ntire23-006`–`040` to training.
- `NH-HAZE2/Train/{hazy,clean}`: 75 real outdoor pairs at 1600x1200. Its `pairs.csv` assigns images `01`–`10` to validation and `11`–`75` to training. The exact duplicate `clean/55(1).png` was removed.
- `augmentations/dualskip_gated_cbam/offline_hazy/`: 2,718 generated hazy training inputs, paired with the original `RESIDE-6K-Outdoor/train/clear` images. Generation settings and source/target mapping are in `config.json` and `manifest.csv`; preview: `augmentations/dualskip_gated_cbam/offline_hazy_preview.png`.
- `RESIDE-6K-Outdoor/augmentations/white_haze_density/`: five offline full-image white-haze density variants (10%, 20%, 30%, 40%, 50%) for each RESIDE training input. `pairs.csv` maps each variant to its original clear target, `config.json` records generation settings, and `preview.png` compares all five levels.
- `depth_maps/{dataset}/...`: optional 16-bit relative-depth PNGs generated from each hazy image by `scripts/generate_depth_maps.py`. Relative paths mirror the hazy-image paths so training/evaluation can load aligned maps. These generated maps are local derived data and are ignored by Git.

The combined RESIDE + NH-Haze experiment uses 2,818 training pairs (RESIDE train 2,718 + NH-HAZE2 train 65 + HD-NH-HAZE train 35) and 297 validation pairs (RESIDE val 282 + NH-HAZE2 val 10 + HD-NH-HAZE val 5). Its test sets are RESIDE-6K Outdoor test (500), O-HAZE (45), and NH-HAZE (55). The new dataset keys accepted by the paired dataset loader are `nhhaze2` and `hdnhhaze`. The source manifests define the NH-Haze splits, while RESIDE keeps its existing manifest.

The original baseline still uses only the RESIDE-6K Outdoor train split and its explicit validation split. Keep all test sets out of checkpoint selection. O-HAZE and NH-HAZE are cross-dataset outdoor evaluations. Split details and pair manifests are kept with each dataset.

The offline fog variants are derived augmentation data; the original RESIDE images are unchanged.

Install Python dependencies, including the Windows CUDA-enabled PyTorch build, using the steps in the project-root [README](../README.md). Dataset files are not installed by pip; place them in the layout above before training or evaluation.


## Optional MCBM input

The MCBM training configurations expect density-map PNG files under data/MCBM/. These external input files are not included in the repository.