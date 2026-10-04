# Methodology and assumptions


| Item | Paper | This implementation |
|---|---|---|
| Backbone | Whisper Small, Table 1 | `openai/whisper-small`, immutable commit recorded |
| ERM, Eq. 1 | Mean utterance ASR risk | Mean token CE per utterance, then mean utterances; not token-weighted HF loss |
| SD, Eq. 2 | `ERM + lambda * squared logit norm` | Full SD objective; normalized mean squared **valid** logits (engineering scale choice), inner lambda 0.06 assumed |
| DRO, Eq. 3 | Maximum group mean risk | Maximum over all observed training groups in each fairness microbatch |
| Standalone IRM | Eq. 4 displays the invariance penalty; standalone details unclear | Standard IRMv1: environment-mean prediction risk plus weighted unbiased pair-product penalty, scalar logits at `w=1` |
| Fusion, Eq. 5 | `lambda_e ERM + lambda_s SD + lambda_d DRO + lambda_i IRM` | Uses **full SD including its ERM term**; outer weights 1, 0.06, 1, 0.01 |
| Learning rate | Fixed 4e-5 | Constant 4e-5 from the first optimizer update; no warmup or decay |
| Adapters | Lightweight, no architecture given | LoRA r16/alpha32/dropout0.05, q/v projections; assumed |
| Sampling/optimizer | Not specified | Natural ERM/SD batches; uniform L1 sampling for fairness methods with pooled-risk importance correction; AdamW, effective batch 32, seed42 |
| Splits | Standard EdAcc splits | Public validation/dev supplies training pool; complete speaker-disjoint holdout for selection; test used for reporting only |
| Group coverage | 26 named evaluation groups | Score every lexical, non-ignored test utterance across all 26 groups; report STM-ignore/nonlexical exclusions explicitly; public dev has fewer matching groups |
| Long audio | Whisper 30-second processing | Exclude unaligned >30s training clips with per-group counts; **retain full long test clips**, native long-form decoding |
| Metrics, Eqs. 6/7 | Equal-group macro-WER, max-min WER | Corpus WER per L1, then equal-group mean and max-min; WER may exceed100% through insertions |
| Transcript cleanup | EdAcc scoring removes special tokens | Exclude `IGNORE_TIME_SEGMENT_IN_SCORING` and nonlexical-only references; strip event tags from speech; preserve complete scorable audio |
| Training target case | Not specified | Lowercase cleaned targets, keeping lexical words and apostrophes |
| Text normalization | Not specified | Remove event tags, lowercase, ASCII punctuation to spaces, collapse whitespace, identical for references/hypotheses |

Eq. 2's inner lambda is not separately reported; choosing 0.06 is an assumption,
distinct from Eq. 5's reported outer weight 0.06. With those choices, Fusion's ERM
coefficient is 1.06 and its normalized SD penalty coefficient is 0.0036.
After run007's penalty-only failure, standalone IRM uses standard IRMv1 prediction
risk plus invariance penalty. Both terms average environments equally. A separate
`irm_penalty_weight=1.0` is a declared starting assumption, independent of Fusion's
lambda_i. The penalty is the mean product over distinct example pairs within each
environment, an unbiased empirical-sampling estimate; it can be negative in a batch
and is not clamped. Fusion retains the supplied paper's Eq.4 squared-mean penalty.
This change resolves the missing predictive objective; its weight is not claimed
to be the authors' setting or a validated optimum.

Standalone DRO requires a physical batch equal to the effective batch32, without
accumulating smaller-batch maxima. This preserves the full-batch maximum group risk.
Standalone IRM also requires physical batch32 and accumulation1, with at least two
independently sampled examples per environment for cross-example products. Fusion retains
its documented microbatch approximation and requires separate review before use.
Every fairness microbatch includes all **observed**
training L1s, but missing dev groups cannot be synthesized. These choices, label
normalization, and long-training exclusions can change the paper comparison.

Selection first measures the zero-update/pretrained adapter on the same complete
holdout. ERM checkpoints must improve both holdout macro-WER and gap over that
control; eligible checkpoints are ranked by macro-WER. Other methods select on
holdout macro-WER. This dual-metric ERM selection is the declared engineering
policy, not a selection rule specified by the paper. If no eligible checkpoint exists,
results explicitly say `pretrained_retained`
and `selected_epoch=0`; that is a failed adaptation, not a successful fine-tune.
Improvement on the holdout does not guarantee improvement on the final test. Optional token-mean
ERM/SD uses token-count weighting across accumulation windows; the default remains
the declared utterance-mean interpretation of Eq. 1.

Table 1's Small Fusion target is macro-WER30.3% / gap45.1%; ERM is32.9% /39.4%.
The paper's Small Fusion **does not improve the gap over ERM**. The final output reports
measured deltas and never forces a favorable result or fills missing runs with demo values.

Each method computes **only its own objective and necessary constituent terms**:
ERM: CE only; SD: CE plus the logit penalty; DRO: group CE risks and their maximum;
IRM: environment CE risks, scalar risk derivatives and their pair-product penalty;
Fusion: all terms in Eq. 5.
No standalone method computes or logs unrelated alternative objectives. Only
Fusion combines the objectives into a training gradient.

The ERM quality goal is a **strict decrease in both macro-WER and min-max
gap versus the measured pretrained baseline**, on the same cleaned 26-group test
and decoding protocol. `erm_goal_assessment.json` records both deltas and a goal
status. `run_status.json` distinguishes completed execution from this quality goal.
A retained zero-update adapter is a failed adaptation; matching the baseline is
not improvement. Test results never choose epochs or change hyperparameters.

Sources: [paper](https://arxiv.org/abs/2510.18374),
[original IRMv1](https://arxiv.org/html/1907.02893v3),
[EdAcc](https://huggingface.co/datasets/edinburghcstr/edacc),
[official EdAcc scoring](https://groups.inf.ed.ac.uk/edacc/leaderboard.html),
[NIST STM ignored regions](https://raw.githubusercontent.com/usnistgov/SCTK/master/doc/infmts.htm),
[Whisper documentation](https://huggingface.co/docs/transformers/v4.46.3/model_doc/whisper)\.
