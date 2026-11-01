# Outdoor Image Dehazing

PyTorch implementation and experiment suite for single-image outdoor dehazing. The repository includes a DehazeFormer-T baseline, model and training ablations, dataset utilities, evaluation scripts, and a Gradio inference demo.

## Project scope

The experiments study supervised dehazing on paired hazy/clear images, including cross-dataset evaluation on real outdoor haze. Implemented comparisons include DehazeFormer-T variants (M1–M9), DEA-Net, SLRNet, and WLD-Net. Data-side experiments include spatial haze augmentation, depth conditioning, and MCBM-based haze synthesis.

Experiment configurations define the data sources, model, loss, optimizer, schedule, and output path. Results depend on those settings and the selected checkpoint; evaluation summaries distinguish pilot results from longer runs.

## Environment

Python 3.10 or newer is required. Install a PyTorch and torchvision build compatible with the local CPU or CUDA environment, then install the project and runtime extras:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The single requirements file installs the core package, Gradio demo, depth-map dependencies, and LPIPS support. Select the PyTorch build for the target machine using the [official installation selector](https://pytorch.org/get-started/locally/).

## Data

Dataset files are not distributed with this repository. Place the datasets under data/ using the directory and pair-manifest conventions documented in [data/README.md](data/README.md).

The baseline configuration uses the RESIDE-6K Outdoor train and validation splits. Other configurations may add NH-HAZE2 or HD-NH-HAZE to training. O-HAZE and NH-HAZE are used for external evaluation. The active data sources and splits are defined by each YAML file in configs/experiment/.

Some experiments also depend on locally generated inputs such as offline haze variants and depth maps. MCBM configurations expect density-map PNGs under data/MCBM/. Preparation scripts and configuration files define the required inputs and paths.

## Train and evaluate

Install the package and dependencies with requirements.txt, then run the baseline:

```bash
python -m dehazing.train --config configs/experiment/baseline.yaml
python -m dehazing.evaluate --checkpoint runs/dehazeformer_t/checkpoints/best.pth --dataset all
```

Use another configuration to run an architecture experiment. For example:

```bash
python -m dehazing.train --config configs/experiment/dea_net.yaml
```

Evaluation supports individual datasets and split selection. Large images are processed with overlapping tiles by default. Evaluator options are available through:

```bash
python -m dehazing.evaluate --help
```

Training logs, metrics, checkpoints, and predictions are written under runs/.

## Gradio demo

Run the application:

```bash
python demo.py
```

The demo lists configured models for which a checkpoint exists in runs/. It can generate a local QR code or use a supplied URL; python demo.py --help lists the options.

## Results and documentation

- [Full-run evaluation results](results/evaluation_comparison_3_models.md)
- [12-epoch pilot results](results/evaluation_pilot12_4_models.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Research scope](docs/RESEARCH_PLAN.md)
- [Training and dataset scripts](scripts/README.md)

The evaluation summaries include dataset sizes, metrics, and measurement settings. Pilot results are screening experiments and are not equivalent to the full-run comparisons. Raw training outputs and model checkpoints are local artifacts and are excluded from version control.

## Repository layout

| Path | Contents |
| --- | --- |
| src/dehazing/ | Training, evaluation, datasets, losses, metrics, and model implementations |
| configs/experiment/ | Dataset and experiment configurations |
| scripts/ | Dataset preparation, training launchers, and analysis utilities |
| data/ | User-provided datasets and generated data |
| results/ | Compact metric summaries, plots, and visualizations |
| docs/ | Research scope and experimental protocols |
| runs/ | Generated logs, checkpoints, metrics, and predictions |

## Attribution

Model implementations correspond to methods documented in the source and experiment configurations. External implementation code is not bundled.