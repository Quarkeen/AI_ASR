# Accent-invariant automatic speech recognition

LoRA fine-tuning of **Whisper Small** on **EdAcc** to compare empirical risk minimization (ERM), spectral decoupling (SD), Group DRO and invariant risk minimization (IRM). Evaluation reports equal-group macro word error rate (WER) and the difference between the highest and lowest group WER across 26 accent/L1 groups.

The reference is Swain et al., *Towards Fair ASR For Second Language Speakers Using Fairness Prompted Finetuning*. This is a research implementation with explicit assumptions, rather than a claim of exact reproduction. See [methodology](docs/methodology.md) for objectives, data exclusions, normalization and checkpoint selection.

## Measured results

All entries use the same 9,079 scorable test utterances and 26 evaluation groups. Lower is better for both columns; the gap is measured in percentage points.

| Experiment | Predictor | Macro-WER (%) | Min-max gap (pp) |
|---|---|---:|---:|
| Control | Pretrained Whisper Small | 24.27 | 26.23 |
| [003](runs/run_003_erm/) | ERM, selected epoch 1 | 20.58 | 36.64 |
| [004](runs/run_004_sd/) | SD, selected epoch 2 | 25.12 | 64.22 |
| [005](runs/run_005_group_dro/) | Group DRO, selected epoch 2 | 21.53 | 31.93 |
| [007](runs/run_007_irm_penalty_only/) | Penalty-only IRM ablation, pretrained fallback | 24.27 | 26.23 |
| [008](runs/run_008_irm/) | Corrected IRMv1, selected epoch 2 | 22.11 | 67.97 |

**No trained method improved both final-test metrics over the control.** Run 007 completed training but retained the pretrained predictor because its trained checkpoints failed holdout selection. Its identical baseline result is not a successful fine-tune. Run 008 includes a separately labelled best-trained result; it uses the same epoch-2 checkpoint as the selected predictor, so its metrics are identical.

Higher experiment numbers mean later experiments. Gaps in numbering preserve original experiment identity after incomplete experiments were removed. [The experiment index](runs/index.json) identifies the latest completed run. Fusion is implemented as an optional objective but has no completed reported experiment.

## Reproduce an experiment

1. Use Python 3.10–3.12 with CUDA-enabled PyTorch. Completed experiments used PyTorch 2.8.0. Install the compatible PyTorch build for your system separately; then install the packages in [requirements.txt](requirements.txt).
2. Open [notebooks/fair_asr.ipynb](notebooks/fair_asr.ipynb). Its dependency cell can install the ASR packages in the notebook kernel. The notebook is self-contained and embeds the canonical implementation.
3. Select methods in `CONFIG["strategies"]`: `ERM`, `SD`, `DRO` or `IRM`. The default is ERM. Each starts from the same pinned pretrained backbone. Choose a fresh output directory when changing settings.
4. Run the cells in order. The data cell downloads EdAcc and prepares persistent features. Initial preparation needs approximately 30 GiB of free disk. Progress bars show training loss, epoch, elapsed time and ETA; inference reports completion and ETA.
5. Collect the generated metrics, per-accent WER, predictions, histories and checkpoint selections. Resume only with an unchanged configuration and dataset. Test scores never select checkpoints.

The completed experiments used learning rate 4e-5, effective batch 32, LoRA rank 16/alpha 32/dropout 0.05, seed 42, a 10-epoch cap and patience 3. Their exact model/dataset commits, software versions, physical batches and scientific settings are retained in each experiment's `config.json`. These curated configurations document historical recipes; the generated notebook's configuration is the supported entry point, and archived results are not live resume directories.

## Repository contents

- [fair_asr_core.py](fair_asr_core.py): objectives, dataset preparation, LoRA training, checkpoint selection and inference.
- [notebooks/fair_asr.ipynb](notebooks/fair_asr.ipynb): readable notebook generated from the same implementation.
- [scripts/build_notebook.py](scripts/build_notebook.py): regenerate the notebook without executing it.
- [scripts/verify_results.py](scripts/verify_results.py): verify stored metrics and unchanged prediction files without loading a model.
- [scripts/wer_aggregation_diagnostic.py](scripts/wer_aggregation_diagnostic.py): supplementary WER aggregation sensitivity analysis.
- [runs/](runs/): completed experiment results and negative ablation evidence.
- [docs/methodology.md](docs/methodology.md): paper correspondence and implementation assumptions.
- [docs/results.md](docs/results.md): result interpretation and known limitations.
- `references/research_paper.pdf`: the supplied source paper, retained locally; PDFs are excluded from version control.


Downloaded audio, feature caches, environments, generated working outputs and model weights are ignored. Measured results, unchanged prediction text and histories remain included. No model weights are bundled. Respect the model and dataset licenses when reusing or redistributing their contents; this repository does not grant additional rights to third-party material.
