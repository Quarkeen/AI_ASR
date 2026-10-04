# Results and interpretation

The completed experiments share one immutable Whisper Small backbone, EdAcc revision, scoring policy and 26-group evaluation set. Per-experiment metrics, group error counts, histories and selections are retained. Predictions are preserved byte-for-byte, including natural spoken text; no references or hypotheses were rewritten during repository cleanup.

The public development/validation split supplies the training pool. After filtering and speaker-disjoint holdout selection there are 4,294 training and 2,402 holdout utterances. Fourteen accent/L1 environments are available for training; all 26 paper groups are evaluated, including 12 groups absent from this training pool. These are English speech groups, not 26 different transcription languages. Of 9,289 source test utterances, 210 ignored or nonlexical references are excluded under the declared scoring policy, leaving 9,079. Long scorable test recordings are retained.

ERM and Group DRO reduce macro-WER relative to the pretrained control but increase the group gap. SD worsens both metrics. The standalone penalty-only IRM ablation trained for three epochs and retained the pretrained predictor because no trained checkpoint improved holdout macro-WER. Corrected IRMv1 adds environment-mean prediction risk, uses an unbiased within-environment pair-product penalty, and selects trained epoch 2 after five epochs. It improves macro-WER while substantially worsening the test gap.

Corrected IRM's Southern British English group has WER 76.60%, including 1,671 insertions over 2,885 reference words. Four severe generation outliers account for approximately 74.4% of that group's word errors. See [error analysis](../runs/run_008_irm/error_analysis.csv). This evidence describes a real evaluation failure; the utterances remain included in all reported results. Training loss or average WER improvement does not establish improved worst-group behavior.

Checkpoint selection uses the complete holdout. ERM requires lower holdout macro-WER and lower gap than the control; other methods rank eligible checkpoints by holdout macro-WER. A retained pretrained checkpoint is labelled explicitly. Final-test metrics were not used to select epochs or tune hyperparameters.

Paper Table 1 reports a different baseline and training outcomes. Adapter architecture, split details, preprocessing and some objective reductions are not fully specified in the paper. This implementation declares its choices and preserves measured outcomes; its results cannot establish exact paper reproduction or attribute differences to a supposedly newer pretrained model.

Each completed experiment contains:

- `config.json`: scientific settings, pinned revisions, precision, package versions and physical batching.
- `data_protocol.json`: split coverage, exclusions and speaker separation checks.
- `measured_results.json`: full-precision metrics, group WER and substitution/deletion/insertion counts.
- `results.csv` and `per_accent_wer.csv`: readable metric tables.
- `<METHOD>/history.json`: optimizer updates, losses, relevant components and holdout scores.
- `<METHOD>/selection.json`: selected epoch, selection policy and predictor role.
- `predictions/*.jsonl`: unchanged reference/hypothesis records, including resumable holdout evaluations for corrected IRM.
- `prediction_checksums.json`: content hashes for the retained prediction files.

The historical configuration files have neutral paths and describe only their completed method. They preserve numerical training settings, while removing device inventory and execution bookkeeping. They are evidence of the completed experiments, not checkpoint-resume manifests. The penalty-only ablation is preserved for interpretation; the canonical implementation uses corrected IRMv1.
