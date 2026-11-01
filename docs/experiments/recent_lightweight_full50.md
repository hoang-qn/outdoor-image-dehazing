# SLRNet and WLD-Net comparison

This experiment evaluates SLRNet and WLD-Net under a shared training recipe on the combined RESIDE-6K Outdoor, NH-HAZE2, and HD-NH-HAZE training splits.

## Protocol

- Training: the train split from each of the three datasets.
- Validation: the corresponding validation splits.
- Final evaluation: RESIDE-6K Outdoor test, O-HAZE, and NH-HAZE.
- Shared settings: seed 42, 256-pixel crops, batch size 8, AdamW, learning rate 2e-4, mixed precision, horizontal flips, and L1 loss.
- Training limit: 50 epochs, with checkpoint selection by validation PSNR.

The model implementations use the package's RGB [0, 1] interface. This shared recipe supports a consistent implementation-level comparison, but does not reproduce the original training schedule or loss for each paper. Reported parameter counts from the original papers are not directly comparable to the measured results unless implementation details also match.

## Run

```bash
python -m dehazing.train --config configs/experiment/slrnet_recent_full50.yaml
python -m dehazing.train --config configs/experiment/wldnet_recent_full50.yaml
```

The experiment configurations define the output paths. Evaluation metrics and the measurement protocol are summarized in [the full-run comparison](../../results/evaluation_comparison_3_models.md).