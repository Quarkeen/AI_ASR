# Completed experiments

See [the main result table](../README.md) and [result interpretation](../docs/results.md). Higher run numbers mean later experiments; [index.json](index.json) lists only completed runs and identifies run 008 as the latest.

Each experiment preserves metrics, scientific configuration, data protocol, training history, checkpoint selection and unchanged predictions. Run 007 is an explicitly labelled negative penalty-only IRM ablation. Run 008 uses corrected IRMv1. No reported trained method achieves both lower macro-WER and a lower min-max gap than the pretrained control.

Use `python3 scripts/verify_results.py` from the repository root to audit the saved result evidence without loading a model or executing a notebook.
