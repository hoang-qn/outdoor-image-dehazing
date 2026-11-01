# Training curves for six runs

The figures summarize training L1 and validation MAE by epoch, and validation SSIM by epoch, for six recorded runs.

## Figures

- [Training L1 and validation MAE](train_l1_validation_mae_6_runs.png) · [SVG](train_l1_validation_mae_6_runs.svg)
- [Training and validation SSIM](train_validation_ssim_6_runs.png) · [SVG](train_validation_ssim_6_runs.svg)

The vertical markers and diamonds identify the epoch with the highest validation PSNR, corresponding to the selected checkpoint. Panels share an epoch axis for readability. A run stopped early is shown only for epochs present in its log.

## Interpretation

train_l1 is the reconstruction L1 component. M2, M3, and M6 include additional loss terms, so this curve is not their total optimization loss. M6 uses MCBM-augmented training data.

The RESIDE-only baseline uses the RESIDE validation split. The other runs use mixed validation data from RESIDE, NH-HAZE2, and HD-NH-HAZE. Batch size and training settings also vary; the plots describe optimization histories and do not constitute a fully controlled ablation.

## Source and regeneration

metrics_source.csv contains the plotted metric series. The plotting scripts are plot_training_curves.py and plot_mcbm_history.py. From the repository root, regenerate the figures with:

```powershell
python results/training_curves_6_runs/plot_training_curves.py
```

The PNG outputs use 300 DPI; SVG versions are provided for scalable use.