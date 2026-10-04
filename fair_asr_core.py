"""Paper-objective ASR implementation, embedded verbatim in the generated notebooks.

Importing this module does not download data, train, or require CUDA.
The notebook builder splits the sections into readable, self-contained notebook cells.
"""
# SECTION: imports_and_contract
import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import string
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
if os.environ.get("FAIR_ASR_HEADLESS") == "1":
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm

IMPLEMENTATION_VERSION = "paper-objectives-v5-selective-loss-dual-metric"
# Generation/data identities stay compatible with the verified baseline. Training
# signatures separately reject the old standalone penalty-only IRM checkpoints.
IRM_UPDATE_VERSION = "irmv1-risk-plus-pair-product-environment-mean-v3"
# Log-Mel features are independent of transcript cleanup; reuse the validated cache.
FEATURE_CACHE_VERSION = "paper-objectives-v2"
PAPER_GROUPS = sorted([
    "Italian", "Ghanain English", "Spanish (Mexican)", "Bulgarian",
    "Vietnamese", "Nigerian English", "Catalan", "Urdu", "Romanian",
    "Jamaican English", "Indian English", "Kenyan English",
    "Mainstream US English", "Spanish", "Irish English", "Mandarin",
    "Filipino", "Southern British English", "Sinhalese", "Indonesian",
    "Hebrew", "Scottish English", "French", "Tagalog", "Lithuanian", "Hindi",
])
PAPER_SMALL = {
    "w/o FT": (67.8, 120.5), "ERM": (32.9, 39.4), "DRO": (38.3, 41.9),
    "SD": (41.4, 44.3), "IRM": (43.8, 53.2), "Fusion": (30.3, 45.1),
}


def default_config():
    """Paper values are distinct from documented engineering choices."""
    return {
        "model_name": "openai/whisper-small",  # Paper Table 1, Small only.
        "model_revision": "main",  # Resolved to an immutable commit before loading.
        "dataset_name": "edinburghcstr/edacc",  # Paper section 3.1.
        "dataset_revision": "main",  # Resolved to an immutable commit.
        "language": "en", "task": "transcribe",
        "lr": 4e-5,  # Paper section 3.2.
        "lambda_e": 1.0, "lambda_d": 1.0,  # Paper section 3.2.
        "lambda_s": 0.06, "lambda_i": 0.01,  # Paper Fusion weights.
        "irm_penalty_weight": 1.0,  # Standalone IRMv1 assumption; NOT Fusion lambda_i.
        "irm_penalty_estimator": "pair_product",  # Unbiased all-distinct-example products.
        "irm_diagnostic_batch_size": 8,
        "sd_inner_lambda": 0.06,  # ASSUMED: Eq. 2's lambda is not separately reported.
        "sd_reduction": "valid_logit_mean",  # ASSUMED normalized squared L2.
        "erm_reduction": "utterance_mean",  # Eq. 1 averages utterance risks.
        "transcript_policy": "edacc_scoring_v1",
        "training_text_case": "lowercase",  # ASSUMED; preserve words, remove annotation tags.
        "sample_rate": 16000, "max_train_audio_sec": 30.0,
        "train_group_policy": "paper_named_groups",  # Explicit dev/test mismatch.
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "lora_targets": ["q_proj", "v_proj"],  # ASSUMED adapter architecture.
        "seed": 42, "weight_decay": 0.01,
        "lr_schedule": "constant", "warmup_ratio": 0.0,  # Paper section 3.2: fixed LR.
        "epoch_cap": 10, "early_stop_patience": 3, "min_wer_improvement": 0.1,
        "erm_require_both_holdout_metrics": True,  # User's success criterion; not a paper setting.
        "min_gap_improvement": 0.0,  # Require a strictly smaller gap than the holdout control.
        "effective_batch_size": 32,  # ASSUMED; autotuning keeps this fixed.
        "candidate_batch_sizes": [4, 8, 16, 32],
        "autotune_repeats": 2, "memory_fraction": 0.88,
        "grad_clip_norm": 1.0,
        "num_workers": min(8, os.cpu_count() or 1),
        "preprocess_workers": min(8, os.cpu_count() or 1),
        "preprocess_batch_size": 32,
        "eval_batch_size": "auto", "candidate_eval_batch_sizes": [8, 16, 32, 64],
        "generation_autotune_tokens": 64, "long_eval_batch_size": 4,
        "holdout_fraction": 0.15, "holdout_examples_per_group": None,  # Use the complete holdout.
        "checkpoint_every_updates": 50,
        "max_new_tokens": 440, "num_beams": 1,
        "long_condition_on_prev_tokens": True,
        "strategies": ["ERM", "SD", "DRO", "IRM", "Fusion"],
        "output_dir": "outputs/fair_asr",  # Fresh path; preserve completed old experiments.
        "cache_dir": "cache/fair_asr",
        "precision": "auto", "gradient_checkpointing": False,
    }


def make_lr_scheduler(optimizer, cfg):
    """Keep the paper LR fixed; reject legacy warmup configurations explicitly."""
    if cfg.get("lr_schedule") != "constant" or cfg.get("warmup_ratio", 0.0) != 0.0:
        raise ValueError("Paper runs require lr_schedule='constant' and warmup_ratio=0.0. "
                         "Old scheduled checkpoints require their original implementation.")
    if not math.isfinite(cfg["lr"]) or cfg["lr"] <= 0:
        raise ValueError("Learning rate must be finite and positive")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_text(text):
    # ASSUMED normalization; same rule for every reference and hypothesis.
    text = re.sub(r"<[^>]+>", " ", str(text)).lower()
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    return re.sub(r"\s+", " ", text).strip()


def clean_edacc_transcript(text):
    """STM ignored regions have no ASR target; event tags are not spoken words."""
    raw = str(text).strip()
    if raw.upper() == "IGNORE_TIME_SEGMENT_IN_SCORING":
        return None
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).strip()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def configure_device(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the experiment. CPU is supported only by the unit tests.")
    cfg["precision"] = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_everything(cfg["seed"])
    for key in ["cache_dir", "output_dir"]:
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)
    print("CPU preprocessing workers:", cfg["preprocess_workers"])
    return torch.device("cuda")


# SECTION: objectives
def validate_irm_config(cfg):
    weight = cfg.get("irm_penalty_weight")
    if weight is None or not math.isfinite(weight) or weight <= 0:
        raise ValueError("Standalone IRMv1 requires an explicit finite positive irm_penalty_weight")
    if cfg.get("irm_penalty_estimator") != "pair_product":
        raise ValueError("Standalone IRMv1 requires the pair_product estimator")
    if not isinstance(cfg.get("irm_diagnostic_batch_size"), int) or cfg["irm_diagnostic_batch_size"] < 1:
        raise ValueError("IRM diagnostic batch size must be a positive integer")


def irm_environment_terms(utterance_risk, utterance_derivative, groups):
    """Equal environment risk and an unbiased all-distinct-pairs penalty estimate.

    For n independent draws, mean_{i != j} d_i*d_j estimates (E[d])**2.
    A finite-sample estimate can be negative. Clamping it would introduce bias.
    This needs >=2 examples per environment and one whole optimizer batch.
    """
    ids = torch.unique(groups, sorted=True)
    risks, means, penalties, variances, counts = [], [], [], [], []
    for gid in ids:
        mask = groups == gid
        d = utterance_derivative[mask]
        n = d.numel()
        if n < 2:
            raise ValueError("IRM pair_product requires at least two examples per environment")
        # Explicit products avoid cancellation in (sum(d)^2-sum(d^2)).
        pairs = torch.triu_indices(n, n, offset=1, device=d.device)
        penalties.append((d[pairs[0]] * d[pairs[1]]).mean())
        risks.append(utterance_risk[mask].mean())
        means.append(d.mean())
        variances.append(d.var(unbiased=True))
        counts.append(n)
    return {"group_ids": ids, "group_risks": torch.stack(risks),
            "group_scale_gradients": torch.stack(means),
            "group_penalty_estimates": torch.stack(penalties),
            "group_derivative_variances": torch.stack(variances),
            "group_sample_counts": torch.tensor(counts, device=groups.device)}


class TokenStatistics(torch.autograd.Function):
    """Memory-bounded CE, SD and exact scalar-IRM derivative with analytic backward.

    IRM is the explicitly assumed logit-scaling seq2seq realization:
    d CE(w*z,y)/dw at w=1 = sum(softmax(z)*z) - z[y].
    Its gradient wrt z is p*(1+z-E_p[z]) - one_hot(y).
    This avoids retaining a second large softmax graph for each accent.
    """

    @staticmethod
    def forward(ctx, logits, labels, chunk_size, statistics):
        if logits.shape[:2] != labels.shape or logits.ndim != 3:
            raise ValueError("Expected logits[B,T,V] and labels[B,T]")
        if torch.any(labels.ne(-100) & ((labels < 0) | (labels >= logits.shape[-1]))):
            raise ValueError("A target token is outside the model vocabulary")
        ctx.save_for_backward(logits, labels)
        ctx.chunk_size = int(chunk_size)
        # Bits: CE=1, squared logits=2, scalar-risk derivative=4.
        # Unrequested outputs stay None, including their gradients in backward.
        if statistics not in range(1, 8):
            raise ValueError("Request at least one valid token statistic")
        ctx.set_materialize_grads(False)
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        zflat, yflat = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        valid = torch.nonzero(yflat.ne(-100), as_tuple=False).flatten()
        outputs = [torch.zeros(labels.numel(), device=logits.device, dtype=dtype)
                   if statistics & bit else None for bit in (1, 2, 4)]
        for start in range(0, valid.numel(), ctx.chunk_size):
            ids = valid[start:start + ctx.chunk_size]
            z, y = zflat[ids].to(dtype), yflat[ids]
            if statistics & (1 | 4):
                centered = z - z.amax(dim=-1, keepdim=True)
                selected = centered.gather(1, y[:, None]).squeeze(1)
            if statistics & 1:
                outputs[0][ids] = torch.logsumexp(centered, dim=-1) - selected
            if statistics & 2:
                outputs[1][ids] = z.square().mean(dim=-1)
            if statistics & 4:
                # CE(w*z) is invariant to a common logit shift, including its
                # scalar derivative. Center before subtracting large expectations
                # to avoid cancellation on real Whisper's offset logits.
                p = torch.softmax(centered, dim=-1)
                outputs[2][ids] = (p * centered).sum(dim=-1) - selected
        return tuple(o.reshape(labels.shape) if o is not None else None for o in outputs)

    @staticmethod
    def backward(ctx, grad_ce, grad_sd, grad_scale):
        logits, labels = ctx.saved_tensors
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        zflat, yflat = logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        valid = torch.nonzero(yflat.ne(-100), as_tuple=False).flatten()
        grad = torch.zeros_like(zflat)
        coefficients = [item.reshape(-1).to(dtype) if item is not None else None
                        for item in (grad_ce, grad_sd, grad_scale)]
        for start in range(0, valid.numel(), ctx.chunk_size):
            ids = valid[start:start + ctx.chunk_size]
            z, y = zflat[ids].to(dtype), yflat[ids]
            a, b, c = [item[ids, None] if item is not None else None for item in coefficients]
            g = torch.zeros_like(z)
            if a is not None:
                p = torch.softmax(z, dim=-1)
            if a is not None:
                g.add_(a * p)
                g.scatter_add_(1, y[:, None], -a)
            if b is not None:
                g.add_(b * (2 * z / logits.shape[-1]))
            if c is not None:
                centered = z - z.amax(dim=-1, keepdim=True)
                p = torch.softmax(centered, dim=-1)
                mean = (p * centered).sum(dim=-1, keepdim=True)
                g.add_(c * p * (1 + centered - mean))
                g.scatter_add_(1, y[:, None], -c)
            grad[ids] = g.to(logits.dtype)
        return grad.reshape_as(logits), None, None, None


def loss_components(logits, labels, groups, cfg, sample_weights=None, *, strategy="ERM"):
    """Compute only the selected objective and its necessary constituent terms."""
    if strategy not in {"ERM", "SD", "DRO", "IRM", "Fusion"}:
        raise ValueError(strategy)
    if labels.ne(-100).sum(dim=1).eq(0).any():
        raise ValueError("Every training utterance must have at least one supervised token")
    need_erm = strategy in {"ERM", "SD", "Fusion"}
    need_sd = strategy in {"SD", "Fusion"}
    need_dro = strategy in {"DRO", "Fusion"}
    need_irm = strategy in {"IRM", "Fusion"}
    if strategy == "IRM":
        validate_irm_config(cfg)
    statistics = int(need_erm or need_dro or strategy == "IRM") | (2 * int(need_sd)) | (4 * int(need_irm))
    ce, sd, scale_gradient = TokenStatistics.apply(logits, labels, 128, statistics)
    lengths = labels.ne(-100).sum(dim=1)
    parts = {}
    if ce is not None:
        utterance_risk = ce.sum(dim=1) / lengths
    if need_erm:
        weights = (torch.ones_like(utterance_risk) if sample_weights is None
                   else sample_weights.to(utterance_risk))
        if cfg["erm_reduction"] == "utterance_mean":
            parts["ERM"] = (utterance_risk * weights).mean()
        elif cfg["erm_reduction"] == "token_mean":
            parts["ERM"] = (ce.sum(dim=1) * weights).sum() / lengths.sum()
        else:
            raise ValueError(cfg["erm_reduction"])
    if need_sd:
        penalty = ((sd.sum(dim=1) / lengths) * weights).mean()
        if cfg["sd_reduction"] == "valid_token_l2_mean":
            penalty = penalty * logits.shape[-1]
        elif cfg["sd_reduction"] != "valid_logit_mean":
            raise ValueError(cfg["sd_reduction"])
        parts["SD_penalty"] = penalty
        parts["SD"] = parts["ERM"] + cfg["sd_inner_lambda"] * penalty
    if need_dro or need_irm:
        if groups.shape != lengths.shape:
            raise ValueError("Require one group ID per utterance")
        unique_groups = torch.unique(groups, sorted=True)
        parts["group_ids"] = unique_groups
    if need_dro:
        risks = torch.stack([utterance_risk[groups == gid].mean() for gid in unique_groups])
        parts["group_risks"] = risks
        parts["DRO"] = risks.max()  # Eq. 3 over groups present in this microbatch.
    if need_irm:
        utterance_derivative = scale_gradient.sum(dim=1) / lengths
        group_scale = torch.stack([
            utterance_derivative[groups == gid].mean()
            for gid in unique_groups
        ])
        parts["group_scale_gradients"] = group_scale
        if strategy == "IRM":
            parts.update(irm_environment_terms(utterance_risk, utterance_derivative, groups))
            parts["IRM_risk"] = parts["group_risks"].mean()
            parts["IRM_penalty"] = parts["group_penalty_estimates"].mean()
            parts["IRM_squared_mean_penalty"] = group_scale.square().mean()  # Diagnostic only.
            parts["IRM"] = parts["IRM_risk"] + cfg["irm_penalty_weight"] * parts["IRM_penalty"]
        else:
            # Fusion continues to use the supplied paper's penalty-only Eq4 term.
            parts["IRM"] = group_scale.square().sum()
    if strategy == "Fusion":
        parts["Fusion"] = (cfg["lambda_e"] * parts["ERM"] + cfg["lambda_s"] * parts["SD"]
                           + cfg["lambda_d"] * parts["DRO"] + cfg["lambda_i"] * parts["IRM"])
    return parts


def objective(logits, labels, groups, strategy, cfg, sample_weights=None):
    if strategy not in {"ERM", "SD", "DRO", "IRM", "Fusion"}:
        raise ValueError(strategy)
    parts = loss_components(logits, labels, groups, cfg, sample_weights, strategy=strategy)
    return parts[strategy], parts


# SECTION: data_and_metrics
def assert_coverage(rows, required, name):
    observed = {int(r["group"]) for r in rows}
    expected = set(required)
    if observed != expected:
        raise ValueError(f"{name}: missing={sorted(expected-observed)}, unexpected={sorted(observed-expected)}")


def speaker_holdout(rows, fraction=0.15, seed=42):
    """Keep singleton-speaker groups in train; never pretend holdout covers them."""
    by_group = defaultdict(set)
    speaker_group = {}
    for row in rows:
        group, speaker = int(row["group"]), str(row["speaker"])
        if speaker in speaker_group and speaker_group[speaker] != group:
            raise ValueError(f"Speaker {speaker} has inconsistent L1 group metadata")
        speaker_group[speaker] = group
        by_group[group].add(speaker)
    rng = np.random.default_rng(seed)
    held = set()
    singleton_groups = []
    for group, speakers in sorted(by_group.items()):
        speakers = sorted(speakers)
        if len(speakers) < 2:
            singleton_groups.append(group)
            continue
        count = min(len(speakers) - 1, max(1, round(len(speakers) * fraction)))
        held.update(rng.choice(speakers, count, replace=False).tolist())
    train = [r for r in rows if str(r["speaker"]) not in held]
    holdout = [r for r in rows if str(r["speaker"]) in held]
    if not train or not holdout:
        raise ValueError("Cannot form a nonempty speaker-disjoint holdout without test leakage")
    assert not ({str(r["speaker"]) for r in train} & {str(r["speaker"]) for r in holdout})
    assert_coverage(train, by_group, "Training after speaker holdout")
    return train, holdout, singleton_groups


def stratified_subset(rows, per_group, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["group"])].append(row)
    rng = np.random.default_rng(seed)
    selected = []
    for group in sorted(grouped):
        ids = rng.permutation(len(grouped[group]))[:per_group]
        selected.extend(grouped[group][i] for i in ids)
    return selected


def compute_group_metrics(predictions, required_groups):
    from jiwer import process_words
    grouped = defaultdict(list)
    for row in predictions:
        if clean_edacc_transcript(row["reference"]) is None or not normalize_text(row["reference"]):
            raise ValueError("Ignored or nonlexical reference reached WER scoring; fix the data protocol")
        grouped[int(row["group"])].append(row)
    if set(grouped) != set(required_groups):
        raise ValueError("Metrics require the exact declared evaluation group set")
    metrics, error_counts = {}, {}
    for group, rows in grouped.items():
        result = process_words(
            [normalize_text(r["reference"]) for r in rows],
            [normalize_text(r["hypothesis"]) for r in rows],
        )
        words = result.hits + result.substitutions + result.deletions
        if words == 0:
            raise ValueError(f"Group {group} has no reference words")
        metrics[group] = 100 * (result.substitutions + result.deletions + result.insertions) / words
        error_counts[group] = {"reference_words": words, "hits": result.hits,
                               "substitutions": result.substitutions,
                               "deletions": result.deletions, "insertions": result.insertions,
                               "utterances": len(rows)}
    values = list(metrics.values())
    return {"per_group_wer": metrics, "macro_wer": float(np.mean(values)),
            "minmax_gap": float(max(values) - min(values)), "groups": len(metrics),
            "per_group_error_counts": error_counts}


def decode_audio(audio):
    import soundfile as sf
    if audio.get("bytes") is not None:
        waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    elif audio.get("path"):
        waveform, sample_rate = sf.read(audio["path"], dtype="float32")
    else:
        raise ValueError("Audio has neither bytes nor a readable path")
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate <= 0 or not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError("Empty or nonfinite audio")
    return waveform, int(sample_rate)


def apply_transcript_protocol(rows, report, processor, cfg, is_training):
    """Clean targets after feature lookup, retaining an auditable exclusion ledger."""
    if cfg["transcript_policy"] != "edacc_scoring_v1":
        raise ValueError("Unsupported transcript policy")
    retained, excluded = [], {name: Counter() for name in [
        "ignored_stm_region", "no_lexical_reference", "cleaned_label_too_long"]}
    stripped_tags = Counter()
    for original in rows:
        raw = original.get("raw_reference", original["reference"])
        text = clean_edacc_transcript(raw)
        if text is None:
            excluded["ignored_stm_region"][PAPER_GROUPS[original["group"]]] += 1
            continue
        stripped_tags.update(re.findall(r"<[^>]+>", raw))
        if not normalize_text(text):
            excluded["no_lexical_reference"][PAPER_GROUPS[original["group"]]] += 1
            continue
        labels = []
        if is_training:
            if cfg["training_text_case"] == "lowercase":
                target = text.lower()
            elif cfg["training_text_case"] == "original":
                target = text
            else:
                raise ValueError("Unsupported training transcript case")
            labels = processor.tokenizer(target).input_ids
            if labels and labels[0] == processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"):
                labels = labels[1:]
            if len(labels) > 448:
                excluded["cleaned_label_too_long"][PAPER_GROUPS[original["group"]]] += 1
                continue
        retained.append({**original, "raw_reference": raw, "reference": text, "labels": labels})
    updated = {**report,
        "cached_feature_rows": len(rows), "kept_rows": len(retained),
        "transcript_policy": cfg["transcript_policy"],
        "transcript_exclusions": {name: dict(counts) for name, counts in excluded.items()},
        "removed_event_tags": dict(stripped_tags),
        "scorable_long_rows": sum(row["duration"] > 30 for row in retained),
        "per_group_kept": dict(Counter(PAPER_GROUPS[row["group"]] for row in retained)),
    }
    print("Transcript cleanup:", json.dumps({key: updated[key] for key in [
        "cached_feature_rows", "kept_rows", "transcript_exclusions", "scorable_long_rows"]}))
    return retained, updated


def feature_cache_key(cfg, training_pool):
    return fingerprint({
        "implementation": FEATURE_CACHE_VERSION,
        "model": cfg["model_revision"], "data": cfg["dataset_revision"],
        "groups": PAPER_GROUPS, "training_pool": training_pool,
        "max_train_sec": cfg["max_train_audio_sec"], "sample_rate": cfg["sample_rate"],
    })[:20]


def cached_features_ready(cfg):
    for pool in ["train", "validation"]:
        directory = Path(cfg["cache_dir"]) / feature_cache_key(cfg, pool)
        ready = True
        for split in [pool, "test"]:
            path = directory / split / "manifest.json"
            if not path.exists():
                ready = False
                break
            manifest = json.loads(path.read_text())
            if not manifest.get("complete") or any(
                not (path.parent / row["feature_file"]).is_file() for row in manifest["rows"]
            ):
                ready = False
                break
        if ready:
            return True
    return False


def prepare_feature_cache(split, split_name, processor, cfg, is_training, cache_key):
    """Keep complete long audio in test; explicitly exclude unaligned long training audio."""
    cache = Path(cfg["cache_dir"]) / cache_key / split_name
    manifest_path = cache / "manifest.json"
    reusable_rows = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["complete"] and all((cache / r["feature_file"]).is_file() for r in manifest["rows"]):
            if not any(manifest["report"].get("long_label_exclusions", {}).values()):
                print(f"Reusing {split_name} feature cache: {len(manifest['rows'])} utterances")
                return apply_transcript_protocol(manifest["rows"], manifest["report"], processor, cfg, is_training)
            # Older caches filtered uncleaned labels. Rebuild metadata and recover
            # those rows, reusing every existing feature file rather than duplicating audio.
            reusable_rows = {row["id"]: row for row in manifest["rows"]}
    cache.mkdir(parents=True, exist_ok=True)
    group_to_id = {name: i for i, name in enumerate(PAPER_GROUPS)}
    required_columns = {"l1", "text", "speaker", "audio"}
    if not required_columns.issubset(split.column_names):
        raise ValueError(f"EdAcc schema missing: {sorted(required_columns-set(split.column_names))}")
    max_tokens = 448  # Whisper-small decoder position limit; checked against model config later.
    report = {"source_rows": len(split), "unmapped_l1": {}, "long_train_exclusions": {},
              "long_label_exclusions": {}, "empty_train_exclusions": {}, "long_test_rows": 0}
    counters = {key: Counter() for key in ["unmapped_l1", "long_train_exclusions",
                                           "long_label_exclusions", "empty_train_exclusions"]}
    rows, durations = [], []

    def process(item):
        index, example = item
        raw = example["l1"]
        if raw not in group_to_id:
            return None, "unmapped_l1", raw
        if f"{split_name}:{index}" in reusable_rows:
            return reusable_rows[f"{split_name}:{index}"], None, raw
        waveform, sr = decode_audio(example["audio"])
        duration = len(waveform) / sr
        if is_training and duration > cfg["max_train_audio_sec"]:
            return None, "long_train_exclusions", raw
        if sr != cfg["sample_rate"]:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=cfg["sample_rate"])
        labels = []  # Target cleanup/tokenization occurs after the independent feature cache.
        if example["speaker"] is None:
            raise ValueError("Missing speaker metadata; cannot check leakage")
        features = processor.feature_extractor(
            waveform, sampling_rate=cfg["sample_rate"], return_tensors="np",
            padding="max_length" if duration <= 30 else "longest",
            truncation=False, return_attention_mask=True,
        )
        feature = features.input_features[0].astype(np.float16)
        frames = min(feature.shape[-1], int(features.attention_mask[0].sum()))
        filename = f"{index:06d}.npy"
        np.save(cache / filename, feature, allow_pickle=False)
        row = {
            "id": f"{split_name}:{index}", "group": group_to_id[raw],
            "speaker": str(example["speaker"]), "reference": str(example["text"]),
            "labels": labels, "duration": float(duration), "frames": frames,
            "feature_file": filename, "feature_path": str((cache / filename).resolve()),
        }
        return row, None, raw

    # Limit queued audio to one batch, rather than materializing the corpus in RAM.
    pbar = tqdm(total=len(split), desc=f"{split_name}: decode/resample/log-Mel", unit="utterance")
    with ThreadPoolExecutor(max_workers=cfg["preprocess_workers"]) as executor:
        for start in range(0, len(split), cfg["preprocess_batch_size"]):
            items = [(i, split[i]) for i in range(start, min(len(split), start + cfg["preprocess_batch_size"]))]
            for row, reason, raw in executor.map(process, items):
                if reason:
                    counters[reason][raw] += 1
                else:
                    rows.append(row)
                    durations.append(row["duration"])
                    report["long_test_rows"] += int(not is_training and row["duration"] > 30)
                pbar.update()
            pbar.set_postfix(kept=len(rows), excluded=sum(sum(c.values()) for c in counters.values()))
    pbar.close()
    for key, counts in counters.items():
        report[key] = dict(counts)
    report["kept_rows"] = len(rows)
    report["duration_percentiles_sec"] = np.percentile(durations, [0, 50, 95, 100]).tolist() if durations else []
    report["per_group_kept"] = dict(Counter(PAPER_GROUPS[r["group"]] for r in rows))
    atomic_json(manifest_path, {"complete": True, "rows": rows, "report": report})
    print(json.dumps(report, indent=2))
    return apply_transcript_protocol(rows, report, processor, cfg, is_training)


def prepare_data(processor, cfg):
    from datasets import Audio, load_dataset
    ds = load_dataset(cfg["dataset_name"], revision=cfg["dataset_revision"],
                      cache_dir=str(Path(cfg["cache_dir"]) / "huggingface"))
    print(ds)
    if "test" not in ds:
        raise ValueError("EdAcc test split is required")
    train_split = "train" if "train" in ds else "validation"
    if train_split not in ds:
        raise ValueError("No train or validation/dev training pool")
    cfg["training_pool"] = train_split
    if train_split == "validation":
        print("PROTOCOL: public EdAcc has no train split; validation/dev is the training pool.")
    for name in {train_split, "test"}:
        ds[name] = ds[name].cast_column("audio", Audio(decode=False))
        for column in ["l1", "accent", "raw_accent"]:
            if column in ds[name].column_names:
                print(name, column, dict(Counter(ds[name][column])))
    train_speakers = {str(s) for s in ds[train_split]["speaker"]}
    test_speakers = {str(s) for s in ds["test"]["speaker"]}
    overlap = sorted(train_speakers & test_speakers)
    if overlap:
        raise ValueError(f"Official training pool/test speaker overlap ({len(overlap)}). Review protocol before training.")
    print(f"Official training pool/test speaker overlap: {len(overlap)}")
    cfg["data_cache_key"] = feature_cache_key(cfg, train_split)
    pool, train_report = prepare_feature_cache(ds[train_split], train_split, processor, cfg, True, cfg["data_cache_key"])
    test, test_report = prepare_feature_cache(ds["test"], "test", processor, cfg, False, cfg["data_cache_key"])
    text_excluded = sum(sum(counts.values()) for counts in test_report["transcript_exclusions"].values())
    if len(test) + text_excluded != len(ds["test"]):
        raise ValueError("Every official test row must be scorable or have an explicit transcript exclusion")
    assert_coverage(test, range(len(PAPER_GROUPS)), "Complete held-out test")
    train, holdout, singleton = speaker_holdout(pool, cfg["holdout_fraction"], cfg["seed"])
    active = sorted({r["group"] for r in train})
    missing = sorted(set(range(len(PAPER_GROUPS))) - set(active))
    holdout_groups = sorted({r["group"] for r in holdout})
    cfg["active_train_groups"] = active
    cfg["holdout_groups"] = holdout_groups
    cfg["data_fingerprint"] = fingerprint({
        "cache": cfg["data_cache_key"], "train": [r["id"] for r in train],
        "holdout": [r["id"] for r in holdout], "test": [r["id"] for r in test],
        "transcript_policy": cfg["transcript_policy"], "training_text_case": cfg["training_text_case"],
        "implementation": IMPLEMENTATION_VERSION,
    })
    protocol = {
        "training_pool": train_split, "train_examples": len(train),
        "holdout_examples": len(holdout), "test_examples": len(test),
        "test_source_examples": len(ds["test"]), "test_transcript_exclusions": text_excluded,
        "training_groups": [PAPER_GROUPS[g] for g in active],
        "unseen_test_groups": [PAPER_GROUPS[g] for g in missing],
        "holdout_groups": [PAPER_GROUPS[g] for g in holdout_groups],
        "singleton_training_groups": [PAPER_GROUPS[g] for g in singleton],
        "speaker_overlap_train_test": overlap,
        "train_exclusions": train_report, "test_retention": test_report,
        "interpretation": "26-group test metrics; early stopping uses only represented holdout groups. "
                          "Unseen test L1s receive no supervised training. Long unaligned training clips "
                          "are excluded. Scoring ignores STM-excluded and nonlexical-only references; "
                          "all complete scorable test clips, including long clips, are transcribed.",
    }
    limit = cfg["holdout_examples_per_group"]
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("holdout_examples_per_group must be None or a positive integer")
    holdout_eval = holdout if limit is None else stratified_subset(holdout, limit, cfg["seed"])
    protocol["selection_holdout_examples"] = len(holdout_eval)
    protocol["selection_holdout_policy"] = "complete" if limit is None else f"up_to_{limit}_per_group"
    atomic_json(Path(cfg["output_dir"]) / "data_protocol.json", protocol)
    print(json.dumps(protocol, indent=2))
    return train, holdout_eval, test


# SECTION: loaders_and_model
class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features = np.load(row["feature_path"], mmap_mode="r", allow_pickle=False)
        return {"features": torch.from_numpy(np.array(features, copy=True)), **row}


class WhisperCollator:
    """Feature extraction happens once in the cache, not every epoch."""
    def __init__(self, pad_token_id):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples):
        if not examples:
            raise ValueError("Empty minibatch")
        width = max(e["features"].shape[-1] for e in examples)
        # Whisper short-form encoder consumes exactly 3000 frames.
        width = max(3000, math.ceil(width / 3000) * 3000)
        feats = torch.zeros(len(examples), examples[0]["features"].shape[0], width, dtype=torch.float16)
        mask = torch.zeros(len(examples), width, dtype=torch.long)
        label_length = max(len(e["labels"]) for e in examples)
        label_length = math.ceil(label_length / 8) * 8
        labels = torch.full((len(examples), label_length), -100, dtype=torch.long)
        for i, example in enumerate(examples):
            feats[i, :, :example["features"].shape[-1]] = example["features"]
            mask[i, :example["frames"]] = 1
            if example["labels"]:
                labels[i, :len(example["labels"])] = torch.tensor(example["labels"])
        return {"input_features": feats, "attention_mask": mask, "labels": labels,
                "groups": torch.tensor([e["group"] for e in examples], dtype=torch.long),
                "rows": [{k: v for k, v in e.items() if k != "features"} for e in examples]}


class EpochBatchSampler(torch.utils.data.Sampler):
    """Deterministic same-device resume; fairness batches sample L1s uniformly.

    ERM/SD see each example exactly once in shuffled order.
    DRO/IRM/Fusion oversample rare groups, with replacement, and use importance
    weights for the pooled ERM/SD terms to retain their empirical-risk meaning.
    """
    def __init__(self, groups, batch_size, seed, epoch, balanced=False, start_batch=0):
        self.groups = np.asarray(groups, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed, self.epoch = int(seed), int(epoch)
        self.balanced, self.start_batch = bool(balanced), int(start_batch)

    def __len__(self):
        return max(0, math.ceil(len(self.groups) / self.batch_size) - self.start_batch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches = math.ceil(len(self.groups) / self.batch_size)
        if not self.balanced:
            order = rng.permutation(len(self.groups)).tolist()
            for batch in range(batches):
                indices = order[batch*self.batch_size:(batch+1)*self.batch_size]
                if batch >= self.start_batch:
                    yield indices
            return
        unique = np.unique(self.groups)
        indices = {g: np.flatnonzero(self.groups == g) for g in unique}
        for batch in range(batches):
            selected_groups = rng.choice(unique, min(len(unique), self.batch_size), replace=False)
            group_sequence = np.resize(selected_groups, self.batch_size)
            rng.shuffle(group_sequence)
            sampled = [int(rng.choice(indices[g])) for g in group_sequence]
            if batch >= self.start_batch:
                yield sampled


def amp_dtype(cfg):
    return torch.bfloat16 if cfg["precision"] == "bf16" else torch.float16


def load_backbone(cfg, trainable=False):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(
        cfg["model_name"], revision=cfg["model_revision"], language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        cfg["model_name"], revision=cfg["model_revision"],
        torch_dtype=amp_dtype(cfg), attn_implementation="sdpa", use_safetensors=True)
    model.generation_config.language = cfg["language"]
    model.generation_config.task = cfg["task"]
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if model.config.max_target_positions != 448:
        raise ValueError("Review label/generation limits for this model")
    if trainable:
        from peft import LoraConfig, get_peft_model
        lora = LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"], target_modules=cfg["lora_targets"],
            bias="none")  # Generic PEFT wrapper; no incompatible text-only TaskType.
        model = get_peft_model(model, lora)
        model.config.use_cache = False
        if cfg["gradient_checkpointing"]:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.print_trainable_parameters()
    model.to("cuda")
    return model, processor


def move_batch(batch, cfg):
    return {
        "input_features": batch["input_features"].to("cuda", dtype=amp_dtype(cfg), non_blocking=True),
        "attention_mask": batch["attention_mask"].to("cuda", non_blocking=True),
        "labels": batch["labels"].to("cuda", non_blocking=True),
        "groups": batch["groups"].to("cuda", non_blocking=True),
    }


def decoder_input_ids(labels, start_token_id, pad_token_id):
    """Whisper's teacher-forcing shift, without allocating an unused HF CE graph."""
    shifted = labels.new_full(labels.shape, pad_token_id)
    shifted[:, 0] = start_token_id
    shifted[:, 1:] = labels[:, :-1]
    return shifted.masked_fill(shifted.eq(-100), pad_token_id)


def forward_logits(model, batch):
    inputs = decoder_input_ids(batch["labels"], model.config.decoder_start_token_id,
                               model.config.pad_token_id)
    return model(input_features=batch["input_features"], decoder_input_ids=inputs,
                 attention_mask=batch["attention_mask"], use_cache=False).logits


def importance_weights(groups, rows, balanced):
    if not balanced:
        return None
    counts = Counter(int(r["group"]) for r in rows)
    weights = torch.zeros(max(counts)+1, device=groups.device, dtype=torch.float32)
    for group, count in counts.items():
        weights[group] = len(counts) * count / len(rows)
    return weights[groups]


def validate_dro_batch_size(batch_size, cfg):
    """Averaging maxima of smaller batches changes the full-batch DRO objective."""
    if batch_size != cfg["effective_batch_size"]:
        raise ValueError("DRO requires one physical effective batch; do not accumulate microbatch maxima")
    if batch_size < len(cfg["active_train_groups"]):
        raise ValueError("DRO batch must represent every observed training group")


def validate_irm_batch_size(batch_size, cfg):
    """One whole optimizer batch with >=2 draws per environment for pair products."""
    if batch_size != cfg["effective_batch_size"]:
        raise ValueError("IRM requires one physical effective batch; do not average squared microbatch gradients")
    if batch_size < 2 * len(cfg["active_train_groups"]):
        raise ValueError("IRM batch must represent every observed training environment at least twice")


def tune_batch_size(model, processor, rows, strategy, cfg):
    """Select the fastest measured batch with memory headroom; never optimize by VRAM alone."""
    model.train()

    collator = WhisperCollator(processor.tokenizer.pad_token_id)
    # Probe longest supervised transcript, rather than an unrealistically tiny batch.
    longest = max(rows, key=lambda r: len(r["labels"]))
    cpu_example = FeatureDataset([longest])[0]
    total_memory = torch.cuda.get_device_properties(0).total_memory
    trials = []
    for batch_size in cfg["candidate_batch_sizes"]:
        if cfg["effective_batch_size"] % batch_size:
            continue
        if strategy in {"DRO", "IRM"} and batch_size != cfg["effective_batch_size"]:
            continue  # Nonlinear group objectives must use the whole optimizer batch.
        if strategy in {"DRO", "IRM", "Fusion"} and batch_size < len(cfg["active_train_groups"]):
            continue  # Every fairness microbatch must represent every observed training L1.
        batch = output = loss = parts = None
        try:
            batch = move_batch(collator([cpu_example] * batch_size), cfg)
            active_ids = torch.tensor(cfg["active_train_groups"], device="cuda", dtype=torch.long)
            batch["groups"] = active_ids[torch.arange(batch_size, device="cuda") % len(active_ids)]
            samples_per_sec = []
            peaks = []
            for repeat in range(cfg["autotune_repeats"] + 1):
                model.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=amp_dtype(cfg)):
                    output = forward_logits(model, batch)
                    loss, parts = objective(output, batch["labels"], batch["groups"], strategy, cfg)
                loss.backward()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if repeat:
                    samples_per_sec.append(batch_size / elapsed)
                    peaks.append(torch.cuda.max_memory_allocated())
                output = loss = parts = None
            peak = max(peaks)
            trial = {"batch_size": batch_size, "examples_per_sec": float(np.median(samples_per_sec)),
                     "safe": peak <= cfg["memory_fraction"] * total_memory}
            trials.append(trial)
            print(strategy, "autotune:", trial)
        except torch.cuda.OutOfMemoryError:
            trials.append({"batch_size": batch_size, "safe": False, "error": "CUDA OOM"})
            print(strategy, "autotune:", batch_size, "does not fit")
            break
        finally:
            model.zero_grad(set_to_none=True)
            batch = output = loss = parts = None
            gc.collect()
            torch.cuda.empty_cache()
    valid = [r for r in trials if r["safe"]]
    if not valid:
        if strategy in {"DRO", "IRM"}:
            raise RuntimeError(f"{strategy}'s full physical effective batch does not fit with memory headroom. "
                               "Review gradient checkpointing; do not average nonlinear microbatch objectives.")
        raise RuntimeError("No candidate fits with memory headroom. Reduce candidate batch sizes or enable gradient checkpointing.")
    selected = max(valid, key=lambda r: r["examples_per_sec"])["batch_size"]
    if strategy == "DRO":
        validate_dro_batch_size(selected, cfg)
    if strategy == "IRM":
        validate_irm_batch_size(selected, cfg)
    atomic_json(Path(cfg["output_dir"]) / f"{strategy}_batch_tuning.json",
                {"trials": trials, "selected_batch_size": selected})
    seed_everything(cfg["seed"])  # Probes must not change training RNG.
    return selected


# SECTION: inference
def prediction_diagnostics(predictions, metrics):
    """Saved-text diagnostics; high repetition alone does not establish an ASR error."""
    def summarize(rows):
        references = [normalize_text(row["reference"]).split() for row in rows]
        hypotheses = [normalize_text(row["hypothesis"]).split() for row in rows]
        ref_count = sum(map(len, references))
        hyp_count = sum(map(len, hypotheses))
        repeated = []
        for words in hypotheses:
            grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
            repeated.append(1 - len(set(grams))/len(grams) if grams else 0.0)
        return {"utterances": len(rows), "reference_words": ref_count,
                "hypothesis_words": hyp_count,
                "hypothesis_to_reference_word_ratio": hyp_count/ref_count if ref_count else None,
                "empty_hypothesis_count": sum(not words for words in hypotheses),
                "mean_repeated_4gram_fraction": sum(repeated)/len(rows),
                "max_hypothesis_words": max(map(len, hypotheses)),
                "mean_hypothesis_words": hyp_count/len(rows)}
    grouped = defaultdict(list)
    for row in predictions:
        grouped[int(row["group"])].append(row)
    result = summarize(predictions)
    result["per_group"] = {}
    for gid, rows in grouped.items():
        count = metrics["per_group_error_counts"].get(gid)
        if count is None:
            count = metrics["per_group_error_counts"][str(gid)]
        result["per_group"][gid] = {
            **summarize(rows),
            **{f"{key}_rate_percent": 100*count[key]/count["reference_words"]
               for key in ("insertions", "deletions", "substitutions")}}
    return result


def irm_teacher_forced_diagnostics(model, processor, rows, cfg):
    """Full holdout CE/confidence/entropy in eval mode, without optimizer updates.

    Measures teacher-forced confidence, not confidence during free decoding.
    Token statistics are chunked to avoid another full-vocabulary FP32 tensor.
    The per-utterance measures are averaged equally within each environment.
    """
    validate_irm_config(cfg)
    if not rows:
        raise ValueError("IRM diagnostics require a nonempty holdout")

    was_training = model.training
    model.eval()
    collator = WhisperCollator(processor.tokenizer.pad_token_id)
    sums = defaultdict(lambda: defaultdict(float))
    size = cfg["irm_diagnostic_batch_size"]
    progress = tqdm(total=len(rows), desc="IRM: validation diagnostics", unit="utterance")
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), size):
                selected = rows[start:start+size]
                batch = move_batch(collator([FeatureDataset([r])[0] for r in selected]), cfg)
                with torch.autocast("cuda", dtype=amp_dtype(cfg)):
                    logits = forward_logits(model, batch)
                ce, _, derivative = TokenStatistics.apply(logits, batch["labels"], 128, 5)
                valid = batch["labels"].ne(-100)
                lengths = valid.sum(1)
                ce_values = ce.sum(1)/lengths
                d_values = derivative.sum(1)/lengths
                entropy = torch.zeros_like(ce)
                confidence, accuracy, logit_std = [torch.zeros_like(ce) for _ in range(3)]
                ids = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
                flat_logits = logits.reshape(-1, logits.shape[-1])
                flat_labels = batch["labels"].reshape(-1)
                for offset in range(0, ids.numel(), 128):
                    chunk = ids[offset:offset+128]
                    z = flat_logits[chunk].float()
                    z = z-z.amax(-1, keepdim=True)
                    logp = torch.log_softmax(z, -1)
                    p = logp.exp()
                    entropy.reshape(-1)[chunk] = -(p*logp).sum(-1)
                    confidence.reshape(-1)[chunk] = p.amax(-1)
                    accuracy.reshape(-1)[chunk] = z.argmax(-1).eq(flat_labels[chunk]).float()
                    logit_std.reshape(-1)[chunk] = z.std(-1, unbiased=False)
                measured = {"ce": ce_values, "scale_derivative": d_values,
                            "entropy_nats": entropy.sum(1)/lengths,
                            "max_token_probability": confidence.sum(1)/lengths,
                            "token_accuracy": accuracy.sum(1)/lengths,
                            "logit_std": logit_std.sum(1)/lengths}
                for index, row in enumerate(selected):
                    target = sums[int(row["group"])]
                    target["utterances"] += 1
                    target["valid_target_tokens"] += int(lengths[index])
                    for key, values in measured.items():
                        value = float(values[index])
                        if not math.isfinite(value):
                            raise FloatingPointError("Nonfinite IRM validation diagnostic")
                        target[key] += value
                    target["scale_derivative_squared"] += float(d_values[index])**2
                progress.update(len(selected))
                del logits, ce, derivative, batch, entropy, confidence, accuracy, logit_std
                del measured, flat_logits, flat_labels, z, p, logp
    finally:
        progress.close()
        model.train(was_training)
    expected = {int(row["group"]) for row in rows}
    if set(sums) != expected:
        raise ValueError("IRM diagnostic environment coverage mismatch")
    per_group = {}
    for gid, total in sums.items():
        n = total["utterances"]
        record = {key: total[key]/n for key in (
            "ce", "scale_derivative", "entropy_nats", "max_token_probability", "token_accuracy", "logit_std")}
        record.update(utterances=int(n), valid_target_tokens=int(total["valid_target_tokens"]))
        record["scale_derivative_variance"] = ((total["scale_derivative_squared"]-total["scale_derivative"]**2/n)/(n-1)
                                                 if n > 1 else None)
        # Moment sums can leave a tiny negative variance from roundoff.
        if record["scale_derivative_variance"] is not None:
            record["scale_derivative_variance"] = max(0.0, record["scale_derivative_variance"])
        per_group[gid] = record
    keys = ("ce", "entropy_nats", "max_token_probability", "token_accuracy", "logit_std")
    return {"mode": "teacher_forced_eval_no_parameter_updates", "groups": len(per_group),
            "utterances": len(rows), "per_group": per_group,
            **{f"macro_{key}": sum(r[key] for r in per_group.values())/len(per_group) for key in keys},
            "squared_environment_mean_derivative": sum(r["scale_derivative"]**2 for r in per_group.values())/len(per_group)}


def tune_inference_batch_size(model, processor, rows, cfg):
    """Measure short-form decoder throughput on duplicated audio; never use test WER for tuning."""
    model.eval()
    sample = max((r for r in rows if r["duration"] <= 30), key=lambda r: r["duration"])
    example = FeatureDataset([sample])[0]
    collator = WhisperCollator(processor.tokenizer.pad_token_id)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    trials = []
    batch = tokens = None
    for size in cfg["candidate_eval_batch_sizes"]:
        try:
            batch = collator([example] * size)
            features = batch["input_features"].to("cuda", dtype=amp_dtype(cfg))
            mask = batch["attention_mask"].to("cuda")
            # Each size gets a warmup, so timing excludes first-call initialization.
            for repeat in range(2):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype(cfg)):
                    tokens = model.generate(input_features=features, attention_mask=mask,
                                            language=cfg["language"], task=cfg["task"],
                                            return_timestamps=False, use_cache=True,
                                            max_new_tokens=cfg["generation_autotune_tokens"],
                                            num_beams=1, do_sample=False)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
            # KV caches grow with output length. Estimate full-length KV reserve explicitly,
            # rather than treating a 64-token probe as the memory cost of 440-token decoding.
            config = model.config
            # Conservatively reserve the entire 440-token self-attention KV allocation
            # in addition to the probe peak, including for early-EOS probe outputs.
            kv_extra = cfg["max_new_tokens"] * size * config.decoder_layers * 2 * config.d_model * 2
            peak = torch.cuda.max_memory_allocated()
            trial = {"batch_size": size, "examples_per_sec": size / elapsed,
                     "safe": peak + max(0, kv_extra) <= cfg["memory_fraction"] * total_memory}
            trials.append(trial)
            print("Inference autotune:", trial)
        except torch.cuda.OutOfMemoryError:
            trials.append({"batch_size": size, "safe": False, "error": "CUDA OOM"})
            break
        finally:
            batch = tokens = features = mask = None
            gc.collect()
            torch.cuda.empty_cache()
    safe = [r for r in trials if r["safe"]]
    if not safe:
        raise RuntimeError("Inference tuning did not find a safe batch. Add smaller candidate_eval_batch_sizes.")
    selected = max(safe, key=lambda r: r["examples_per_sec"])["batch_size"]
    atomic_json(Path(cfg["output_dir"]) / "inference_batch_tuning.json",
                {"trials": trials, "selected_batch_size": selected})
    return selected


def evaluation_identity(cfg, label, model_signature):
    return fingerprint({
        "implementation": IMPLEMENTATION_VERSION, "label": label, "model": model_signature,
        "model_revision": cfg["model_revision"], "dataset": cfg.get("data_fingerprint"),
        "max_new_tokens": cfg["max_new_tokens"], "num_beams": cfg["num_beams"],
        "long_decode": "total_decoder_budget_448",
        "long_condition_on_prev_tokens": cfg["long_condition_on_prev_tokens"],
        "normalization": "remove_event_tags_lower_ascii_punctuation_to_space",
    })


def generation_limits(cfg, is_long, max_target_positions=448):
    # A fixed new-token budget plus previous-segment tokens can exceed Whisper's
    # 448-position decoder limit. Native long-form generation must bound total
    # decoder length and let each segment use the remaining budget.
    if is_long:
        return {"max_length": max_target_positions, "max_new_tokens": None,
                "condition_on_prev_tokens": cfg["long_condition_on_prev_tokens"]}
    return {"max_new_tokens": cfg["max_new_tokens"], "condition_on_prev_tokens": False}


def evaluate(model, processor, rows, cfg, required_groups, label="eval", cache_signature=None):
    """Batched full-utterance decoding, including native Whisper long-form decoding."""

    source_rows = {row["id"]: row for row in rows}
    if len(source_rows) != len(rows):
        raise ValueError("Evaluation requires unique utterance IDs")
    model.eval()
    cache_path = None
    predictions = {}
    if cache_signature is not None:
        cache_path = Path(cfg["output_dir"]) / "predictions" / f"{label}.jsonl"
        identity = evaluation_identity(cfg, label, cache_signature)
        metadata_path = cache_path.with_suffix(".metadata.json")
        if cache_path.exists():
            if not metadata_path.exists() or json.loads(metadata_path.read_text())["identity"] != identity:
                raise ValueError(f"Stale predictions at {cache_path}; use a new output directory.")
            lines = cache_path.read_text().splitlines()
            for index, line in enumerate(lines):
                try:
                    prediction = json.loads(line)
                except json.JSONDecodeError:
                    if index != len(lines)-1:
                        raise
                    # Repair only a torn final append after an interrupted run.
                    cache_path.write_text("\n".join(lines[:index]) + ("\n" if index else ""))
                    break
                if prediction["id"] in predictions:
                    raise ValueError("Duplicate utterance in prediction cache")
                predictions[prediction["id"]] = prediction
        else:
            atomic_json(metadata_path, {"identity": identity})
    wanted = {r["id"] for r in rows}
    if set(predictions) - wanted:
        raise ValueError("Prediction cache contains utterances outside this evaluation")
    for key, prediction in predictions.items():
        if any(prediction[field] != source_rows[key][field] for field in ("group", "reference")):
            raise ValueError("Cached prediction reference/group differs from the requested evaluation")
    pending = [r for r in rows if r["id"] not in predictions]
    short = sorted([r for r in pending if r["duration"] <= 30], key=lambda r: r["duration"])
    long = sorted([r for r in pending if r["duration"] > 30], key=lambda r: r["duration"])
    collator = WhisperCollator(processor.tokenizer.pad_token_id)
    started = time.perf_counter()
    progress = tqdm(total=len(rows), initial=len(predictions), desc=f"{label}: inference", unit="utterance")
    was_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        for sequence, batch_size, is_long in (
            (short, cfg["eval_batch_size"], False),
            (long, cfg["long_eval_batch_size"], True),
        ):
            cursor = 0
            while cursor < len(sequence):
                selected = sequence[cursor:cursor + batch_size]
                batch = collator([FeatureDataset([r])[0] for r in selected])
                tokens = None
                try:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype(cfg)):
                        tokens = model.generate(
                            input_features=batch["input_features"].to("cuda", dtype=amp_dtype(cfg)),
                            attention_mask=batch["attention_mask"].to("cuda"),
                            language=cfg["language"], task=cfg["task"],
                            return_timestamps=is_long,
                            **generation_limits(cfg, is_long, model.config.max_target_positions),
                            num_beams=cfg["num_beams"], do_sample=False,
                            use_cache=True,
                        )
                    hypotheses = processor.batch_decode(tokens, skip_special_tokens=True)
                except torch.cuda.OutOfMemoryError:
                    if batch_size <= 1:
                        raise
                    batch_size = max(1, batch_size // 2)
                    batch = tokens = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    tqdm.write(f"{label}: reducing inference batch to {batch_size} after OOM")
                    continue
                new = [{"id": r["id"], "group": r["group"], "speaker": r["speaker"],
                        "duration": r["duration"], "reference": r["reference"], "hypothesis": hyp}
                       for r, hyp in zip(selected, hypotheses)]
                if len(new) != len(selected):
                    raise ValueError("Decoder output count does not match batch size")
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    with cache_path.open("a") as stream:
                        for prediction in new:
                            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                        stream.flush()
                predictions.update({r["id"]: r for r in new})
                batch = tokens = None
                cursor += len(selected)
                progress.update(len(selected))
                progress.set_postfix(batch=batch_size, mode="long" if is_long else "short",
                                     elapsed=f"{time.perf_counter()-started:.0f}s")
    finally:
        progress.close()
        model.config.use_cache = was_cache
    if set(predictions) != wanted:
        raise ValueError("Evaluation did not transcribe every requested utterance")
    result = compute_group_metrics(list(predictions.values()), required_groups)
    result["prediction_diagnostics"] = prediction_diagnostics(list(predictions.values()), result)
    result["evaluation_rows_fingerprint"] = fingerprint([
        (row["id"], row["group"], row["reference"]) for row in sorted(rows, key=lambda r: r["id"])
    ])
    result["inference_seconds_this_session"] = time.perf_counter() - started
    return result


# SECTION: checkpoints_and_training
def checkpoint_improves(strategy, metrics, baseline, best, cfg):
    """Select on holdout only; ERM must beat its control in both requested metrics."""
    for record in (metrics, baseline, best):
        if any(not math.isfinite(record[key]) for key in ("macro_wer", "minmax_gap")):
            raise ValueError("Nonfinite holdout metric cannot select a checkpoint")
    if strategy == "ERM" and cfg["erm_require_both_holdout_metrics"]:
        if not (metrics["macro_wer"] < baseline["macro_wer"] - cfg["min_wer_improvement"]
                and metrics["minmax_gap"] < baseline["minmax_gap"] - cfg["min_gap_improvement"]):
            return False
    return metrics["macro_wer"] < best["macro_wer"] - cfg["min_wer_improvement"]


def erm_goal_assessment(results):
    """Completion and measured improvement are different outcomes; never force scores."""
    if not {"w/o FT", "ERM"}.issubset(results):
        return {"status": "pending", "goal_met": False}
    baseline, erm = results["w/o FT"], results["ERM"]
    if (baseline.get("groups") != 26 or erm.get("groups") != 26
            or not baseline.get("evaluation_rows_fingerprint")
            or baseline["evaluation_rows_fingerprint"] != erm.get("evaluation_rows_fingerprint")):
        raise ValueError("ERM goal requires the same complete 26-group evaluation as baseline")
    macro_delta = erm["macro_wer"] - baseline["macro_wer"]
    gap_delta = erm["minmax_gap"] - baseline["minmax_gap"]
    selection = erm.get("selection", {})
    adapted = selection.get("status") == "finetuned_selected" and selection.get("selected_epoch", 0) > 0
    met = bool(adapted and macro_delta < 0 and gap_delta < 0)
    return {
        "status": "met" if met else "not_met", "goal_met": met,
        "baseline": {key: baseline[key] for key in ("macro_wer", "minmax_gap")},
        "erm": {key: erm[key] for key in ("macro_wer", "minmax_gap")},
        "macro_delta_vs_baseline": macro_delta, "gap_delta_vs_baseline": gap_delta,
        "macro_improved": macro_delta < 0, "gap_improved": gap_delta < 0,
        "finetuned_checkpoint_selected": bool(adapted), "test_used_for_selection": False,
        "interpretation": "Both deltas must be negative on the same cleaned 26-group test. "
                          "Retaining the pretrained adapter is not a successful fine-tune.",
    }


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}


def restore_adapter(model, state):
    from peft import set_peft_model_state_dict
    expected = adapter_state(model)
    if set(expected) != set(state) or any(expected[key].shape != state[key].shape for key in expected):
        raise ValueError("Adapter key/shape mismatch; refuse a partial checkpoint restore")
    result = set_peft_model_state_dict(model, state)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")


def save_adapter_checkpoint(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(adapter_state(model), temporary)
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapter_change_from_control(model, control_path):
    control = torch.load(control_path, map_location="cpu", weights_only=True)
    current = adapter_state(model)
    if set(control) != set(current):
        raise ValueError("Adapter diagnostic key mismatch")
    squared, maximum, changed = 0.0, 0.0, 0
    for key, value in current.items():
        if value.shape != control[key].shape:
            raise ValueError("Adapter diagnostic shape mismatch")
        difference = value.float()-control[key].float()
        squared += float(difference.double().square().sum())
        maximum = max(maximum, float(difference.abs().max()))
        changed += int(torch.count_nonzero(difference) > 0)
    return {"delta_l2": math.sqrt(squared), "max_abs_change": maximum,
            "changed_tensors": changed, "adapter_tensors": len(current)}


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_training_state(path, model, optimizer, scheduler, scaler, state, signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({
        "signature": signature, "adapter": adapter_state(model),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "state": state, "rng": rng_state(),
    }, temporary)
    os.replace(temporary, path)


def train_strategy(strategy, train_rows, holdout_rows, cfg):
    if cfg.get("lr_schedule") != "constant" or cfg.get("warmup_ratio", 0.0) != 0.0:
        raise ValueError("This implementation requires the paper's constant-LR configuration")
    if cfg["erm_reduction"] == "token_mean" and strategy in {"DRO", "IRM", "Fusion"}:
        raise ValueError("The token-mean alternative is supported for naturally sampled ERM/SD only; "
                         "group-balanced token-risk importance weighting needs a separate protocol")
    if strategy == "IRM":
        validate_irm_config(cfg)
    seed_everything(cfg["seed"])
    model, processor = load_backbone(cfg, trainable=True)
    output_dir = Path(cfg["output_dir"]) / strategy
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "resume.pt"
    best_path = output_dir / "best_adapter.pt"
    best_trained_path = output_dir / "best_trained_adapter.pt"
    control_path = output_dir / "checkpoints/epoch_000_adapter.pt"
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path.exists() else None
    if saved is None:
        batch_size = tune_batch_size(model, processor, train_rows, strategy, cfg)
    else:
        batch_size = saved["state"]["batch_size"]
    if strategy == "DRO":
        validate_dro_batch_size(batch_size, cfg)
    if strategy == "IRM":
        validate_irm_batch_size(batch_size, cfg)
    accumulation = cfg["effective_batch_size"] // batch_size
    n_batches = math.ceil(len(train_rows) / batch_size)
    updates_per_epoch = math.ceil(n_batches / accumulation)
    total_updates = updates_per_epoch * cfg["epoch_cap"]
    signature_payload = {
        "implementation": IMPLEMENTATION_VERSION, "strategy": strategy,
        "configuration": cfg, "batch_size": batch_size, "accumulation": accumulation,
    }
    if strategy == "DRO":
        signature_payload["dro_update_version"] = "all-groups-one-physical-batch-v1"
    if strategy == "IRM":
        signature_payload["irm_update_version"] = IRM_UPDATE_VERSION
    signature = fingerprint(signature_payload)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer_options = dict(lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=cfg["weight_decay"])
    try:
        optimizer = torch.optim.AdamW(trainable, fused=True, **optimizer_options)
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(trainable, **optimizer_options)
    scheduler = make_lr_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    state = {"epoch": 0, "next_batch": 0, "updates": 0, "batch_size": batch_size,
             "best_macro_wer": None, "best_epoch": None, "baseline_holdout": None,
             "best_holdout": None, "patience": 0, "finished": False, "history": []}
    if strategy == "IRM":
        state.update(best_trained_epoch=None, best_trained_holdout=None, best_trained_updates=None,
                     best_trained_macro_wer=None)
    if saved is not None:
        if saved["signature"] != signature:
            raise ValueError("Checkpoint/data/configuration mismatch. Use a new output directory; do not silently resume.")
        restore_adapter(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        restore_rng(saved["rng"])
        saved = None
        print(strategy, "resumed at epoch", state["epoch"]+1, "next batch", state["next_batch"])
    balanced = strategy in {"DRO", "IRM", "Fusion"}
    holdout_groups = {r["group"] for r in holdout_rows}
    if saved is None and state["baseline_holdout"] is None:
        # Fresh LoRA has a zero update, so this is the pretrained control on the
        # SAME selection rows. Never select a worse first epoch unconditionally.
        if strategy == "IRM":
            control_hash = save_adapter_checkpoint(control_path, model)
            baseline = evaluate(model, processor, holdout_rows, cfg, holdout_groups,
                                label="IRM_holdout_epoch_000", cache_signature=control_hash)
            baseline["teacher_forced_diagnostics"] = irm_teacher_forced_diagnostics(model, processor, holdout_rows, cfg)
            atomic_json(output_dir / "validation_epoch_000.json", {
                "checkpoint_epoch": 0, "checkpoint_updates": 0, "checkpoint_sha256": control_hash,
                "predictor_role": "pretrained_control", "metrics": baseline})
        else:
            baseline = evaluate(model, processor, holdout_rows, cfg, holdout_groups,
                                label=f"{strategy} holdout before training")
        state["baseline_holdout"] = state["best_holdout"] = baseline
        state["best_macro_wer"], state["best_epoch"] = baseline["macro_wer"], 0
        state["history"].append({"epoch": 0, "update": 0,
                                 "holdout_macro_wer": baseline["macro_wer"],
                                 "holdout_minmax_gap": baseline["minmax_gap"],
                                 "holdout_groups": len(holdout_groups), "pretrained_control": True})
        if strategy == "IRM":
            state["history"][-1]["teacher_forced_diagnostics"] = baseline["teacher_forced_diagnostics"]
        temporary = best_path.with_suffix(".tmp")
        torch.save(adapter_state(model), temporary)
        os.replace(temporary, best_path)
        save_training_state(checkpoint_path, model, optimizer, scheduler, scaler, state, signature)
        atomic_json(output_dir / "history.json", state["history"])
    collator = WhisperCollator(processor.tokenizer.pad_token_id)
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()  # Report training peak separately from discarded tuning probes.
    progress = tqdm(total=total_updates, initial=state["updates"],
                    desc=f"{strategy}: optimizer updates", unit="update")
    checkpoint_rng = rng_state()
    checkpoint_state = {**state, "history": list(state["history"])}
    try:
        while state["epoch"] < cfg["epoch_cap"] and not state["finished"]:
            epoch, start_batch = state["epoch"], state["next_batch"]
            sampler = EpochBatchSampler(
                [r["group"] for r in train_rows], batch_size, cfg["seed"], epoch,
                balanced=balanced, start_batch=start_batch)
            token_mean = not balanced and cfg["erm_reduction"] == "token_mean"
            window_indices = list(EpochBatchSampler(
                [r["group"] for r in train_rows], batch_size, cfg["seed"], epoch
            )) if token_mean else None
            loader = torch.utils.data.DataLoader(
                FeatureDataset(train_rows), batch_sampler=sampler, collate_fn=collator,
                num_workers=cfg["num_workers"], pin_memory=True,
                persistent_workers=cfg["num_workers"] > 0,
                **({"prefetch_factor": 2} if cfg["num_workers"] else {}),
                generator=torch.Generator().manual_seed(cfg["seed"] + epoch),
            )
            model.train()

            optimizer.zero_grad(set_to_none=True)
            window_logs, window_loss, window_examples = [], 0.0, 0
            epoch_started = time.perf_counter()
            window_started = epoch_started
            for batch_index, cpu_batch in enumerate(loader, start=start_batch):
                if strategy in {"DRO", "IRM"}:
                    if (len(cpu_batch["rows"]) != cfg["effective_batch_size"] or
                            set(cpu_batch["groups"].tolist()) != set(cfg["active_train_groups"])):
                        raise ValueError(f"{strategy} requires a full physical batch with every observed training group")
                    if strategy == "IRM" and min(Counter(cpu_batch["groups"].tolist()).values()) < 2:
                        raise ValueError("IRM requires at least two draws per environment")
                batch = move_batch(cpu_batch, cfg)
                window_start = (batch_index // accumulation) * accumulation
                window_end = min(n_batches, window_start + accumulation)
                window_size = window_end - window_start
                # Natural shuffled final batch may be short: weight utterances exactly.
                window_count = min(len(train_rows), window_end*batch_size) - window_start*batch_size
                utterance_coefficient = len(cpu_batch["rows"])/window_count
                if token_mean:
                    # Token-mean CE needs token-count weighting across the entire
                    # accumulation window, rather than averaging microbatch means.
                    indices = window_indices[window_start:window_end]
                    window_tokens = sum(len(train_rows[i]["labels"]) for ids in indices for i in ids)
                    coefficient = int(cpu_batch["labels"].ne(-100).sum()) / window_tokens
                else:
                    coefficient = 1/window_size if balanced else len(cpu_batch["rows"])/window_count
                # Only Fusion needs pooled-risk importance correction among
                # balanced methods. Standalone DRO/IRM do not use pooled ERM.
                weights = importance_weights(batch["groups"], train_rows, strategy == "Fusion")
                with torch.autocast("cuda", dtype=amp_dtype(cfg)):
                    output = forward_logits(model, batch)
                    loss, components = objective(
                        output, batch["labels"], batch["groups"], strategy, cfg, weights)
                    if token_mean and strategy == "SD":
                        # SD's CE is token-weighted, but its declared logit
                        # regularizer is utterance-weighted. Reduce them separately.
                        loss = (components["ERM"] * coefficient + cfg["sd_inner_lambda"] *
                                components["SD_penalty"] * utterance_coefficient)
                        coefficient = 1.0
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{strategy}: nonfinite objective at batch {batch_index}")
                scaler.scale(loss * coefficient).backward()
                window_loss += float(loss.detach()) * coefficient
                window_examples += len(cpu_batch["rows"])
                component_values = {k: float(v.detach()) for k, v in components.items()
                                    if k in {"ERM", "SD", "SD_penalty", "DRO", "IRM", "Fusion",
                                             "IRM_risk", "IRM_penalty", "IRM_squared_mean_penalty"}}
                microbatch_log = {
                    "strategy": strategy, "objective": component_values[strategy],
                    "examples": len(cpu_batch["rows"]), "accumulation_weight": coefficient,
                    "components": component_values,
                }
                if token_mean and strategy == "SD":
                    microbatch_log["ce_accumulation_weight"] = int(cpu_batch["labels"].ne(-100).sum()) / window_tokens
                    microbatch_log["penalty_accumulation_weight"] = utterance_coefficient
                if strategy == "IRM":
                    microbatch_log["irm_penalty_weight"] = cfg["irm_penalty_weight"]
                    microbatch_log["irm_penalty_estimator"] = cfg["irm_penalty_estimator"]
                for key in ("group_risks", "group_scale_gradients", "group_penalty_estimates",
                            "group_derivative_variances", "group_sample_counts"):
                    if key in components:
                        microbatch_log[key] = {str(int(g)): float(v) for g, v in zip(
                            components["group_ids"].detach().cpu(), components[key].detach().cpu())}
                window_logs.append(microbatch_log)
                output = loss = components = batch = None
                if batch_index + 1 != window_end:
                    continue
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable, cfg["grad_clip_norm"], error_if_nonfinite=not scaler.is_enabled())
                previous_scale = scaler.get_scale()
                used_lr = optimizer.param_groups[0]["lr"]
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.is_enabled() and scaler.get_scale() < previous_scale
                optimizer.zero_grad(set_to_none=True)
                if not skipped:
                    scheduler.step()
                    state["updates"] += 1
                    progress.update()
                state["next_batch"] = batch_index + 1
                record = {"epoch": epoch+1, "update": state["updates"],
                          "loss": window_loss, "grad_norm": float(grad_norm) if torch.isfinite(grad_norm) else None,
                          "lr": used_lr, "next_lr": scheduler.get_last_lr()[0], "skipped_update": skipped,
                          "elapsed_seconds": time.perf_counter()-start_time,
                          "examples_per_sec": window_examples / (time.perf_counter()-window_started),
                          "microbatches": window_logs}
                record["gradient_clipped"] = bool(float(grad_norm) > cfg["grad_clip_norm"])
                record["gradient_clip_limit"] = cfg["grad_clip_norm"]
                state["history"].append(record)
                progress.set_postfix(epoch=f"{epoch+1}/{cfg['epoch_cap']}", loss=f"{window_loss:.4f}",
                                     lr=f"{record['lr']:.2e}", epoch_s=f"{time.perf_counter()-epoch_started:.0f}",
                                     ex_s=f"{record['examples_per_sec']:.1f}",
                                     **({"risk": f"{window_logs[0]['components']['IRM_risk']:.4f}",
                                         "penalty": f"{window_logs[0]['components']['IRM_penalty']:.4f}"}
                                        if strategy == "IRM" else {}))
                checkpoint_state = {**state, "history": list(state["history"])}
                checkpoint_rng = rng_state()
                if state["updates"] % cfg["checkpoint_every_updates"] == 0:
                    save_training_state(checkpoint_path, model, optimizer, scheduler, scaler, state, signature)
                    atomic_json(output_dir / "history.json", state["history"])
                window_logs, window_loss, window_examples = [], 0.0, 0
                window_started = time.perf_counter()
            # Validation never uses final test labels for model selection.
            if strategy == "IRM":
                epoch_path = output_dir / f"checkpoints/epoch_{epoch+1:03d}_adapter.pt"
                epoch_hash = save_adapter_checkpoint(epoch_path, model)
                metrics = evaluate(model, processor, holdout_rows, cfg, holdout_groups,
                                   label=f"IRM_holdout_epoch_{epoch+1:03d}", cache_signature=epoch_hash)
                metrics["teacher_forced_diagnostics"] = irm_teacher_forced_diagnostics(model, processor, holdout_rows, cfg)
                metrics["adapter_change_from_pretrained"] = adapter_change_from_control(model, control_path)
                if metrics["adapter_change_from_pretrained"]["changed_tensors"] == 0:
                    raise RuntimeError("IRM made optimizer updates but no adapter parameter changed")
                metrics["diagnostic_flags"] = {
                    "holdout_wer_worsened": metrics["macro_wer"] > state["baseline_holdout"]["macro_wer"],
                    "holdout_ce_worsened": metrics["teacher_forced_diagnostics"]["macro_ce"] >
                        state["baseline_holdout"]["teacher_forced_diagnostics"]["macro_ce"]}
                atomic_json(output_dir / f"validation_epoch_{epoch+1:03d}.json", {
                    "checkpoint_epoch": epoch+1, "checkpoint_updates": state["updates"],
                    "checkpoint_sha256": epoch_hash, "predictor_role": "trained_IRMv1_adapter", "metrics": metrics})
                if state["best_trained_macro_wer"] is None or metrics["macro_wer"] < state["best_trained_macro_wer"]:
                    shutil.copy2(epoch_path, best_trained_path)
                    state.update(best_trained_macro_wer=metrics["macro_wer"], best_trained_epoch=epoch+1,
                                 best_trained_holdout=metrics, best_trained_updates=state["updates"])
            else:
                metrics = evaluate(model, processor, holdout_rows, cfg, holdout_groups,
                                   label=f"{strategy} holdout epoch {epoch+1}")
            macro = metrics["macro_wer"]
            improved = checkpoint_improves(strategy, metrics, state["baseline_holdout"],
                                           state["best_holdout"], cfg)
            if improved:
                state["best_macro_wer"], state["patience"] = macro, 0
                state["best_epoch"], state["best_holdout"] = epoch+1, metrics
                temporary = best_path.with_suffix(".tmp")
                torch.save(adapter_state(model), temporary)
                os.replace(temporary, best_path)
            else:
                state["patience"] += 1
            state["history"].append({"epoch": epoch+1, "update": state["updates"],
                                     "holdout_macro_wer": macro, "holdout_minmax_gap": metrics["minmax_gap"],
                                     "holdout_groups": len(holdout_groups)})
            if strategy == "IRM":
                state["history"][-1].update(
                    teacher_forced_diagnostics=metrics["teacher_forced_diagnostics"],
                    prediction_diagnostics=metrics["prediction_diagnostics"],
                    adapter_change_from_pretrained=metrics["adapter_change_from_pretrained"],
                    diagnostic_flags=metrics["diagnostic_flags"])
            tqdm.write(f"{strategy}: epoch {epoch+1}, holdout macro-WER {macro:.2f}% "
                       f"gap {metrics['minmax_gap']:.2f} points; "
                       f"over {len(holdout_groups)} groups; best={state['best_macro_wer']:.2f}%, "
                       f"patience={state['patience']}/{cfg['early_stop_patience']}")
            state["epoch"] += 1
            state["next_batch"] = 0
            state["finished"] = (state["patience"] >= cfg["early_stop_patience"]
                                 or state["epoch"] >= cfg["epoch_cap"])
            checkpoint_state = {**state, "history": list(state["history"])}
            checkpoint_rng = rng_state()
            save_training_state(checkpoint_path, model, optimizer, scheduler, scaler, state, signature)
            atomic_json(output_dir / "history.json", state["history"])
    except BaseException:
        # Discard an incomplete accumulation window and replay it on resume.
        optimizer.zero_grad(set_to_none=True)
        restore_rng(checkpoint_rng)
        save_training_state(checkpoint_path, model, optimizer, scheduler, scaler, checkpoint_state, signature)
        raise
    finally:
        progress.close()
    if not best_path.exists():
        raise RuntimeError("No validation-selected adapter was produced")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    restore_adapter(model, best)
    model.save_pretrained(output_dir / "best_adapter")
    processor.save_pretrained(output_dir / "best_adapter")
    selection = {
        "selected_epoch": state["best_epoch"],
        "status": "pretrained_retained" if state["best_epoch"] == 0 else "finetuned_selected",
        "baseline_holdout": state["baseline_holdout"], "selected_holdout": state["best_holdout"],
        "selection_metric": "holdout_macro_wer", "test_used_for_selection": False,
        "gap_used_for_selection": strategy == "ERM" and cfg["erm_require_both_holdout_metrics"],
        "eligibility": "both_holdout_metrics_below_pretrained" if
            strategy == "ERM" and cfg["erm_require_both_holdout_metrics"] else "macro_wer_below_pretrained",
    }
    if strategy == "IRM":
        if state["best_trained_epoch"] is None or not best_trained_path.exists():
            raise RuntimeError("IRM has no separately retained trained checkpoint")
        selection.update(
            objective="environment_mean_prediction_risk_plus_weighted_pair_product_penalty",
            irm_penalty_weight=cfg["irm_penalty_weight"], irm_penalty_estimator=cfg["irm_penalty_estimator"],
            irm_update_version=IRM_UPDATE_VERSION,
            best_trained_epoch=state["best_trained_epoch"], best_trained_holdout=state["best_trained_holdout"],
            best_trained_updates=state["best_trained_updates"],
            best_trained_sha256=hashlib.sha256(best_trained_path.read_bytes()).hexdigest(),
            total_optimizer_updates=state["updates"],
            selected_predictor_role="pretrained_fallback" if state["best_epoch"] == 0 else "selected_trained_checkpoint")
        print("IRM selected checkpoint:", selection["selected_predictor_role"], "epoch", state["best_epoch"],
              "; separately retained best trained epoch", state["best_trained_epoch"])
    atomic_json(output_dir / "selection.json", selection)
    signature_hash = hashlib.sha256(best_path.read_bytes()).hexdigest()
    # Release training/optimizer state before full-test inference.
    del optimizer, scheduler, scaler, trainable, best
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    return model, processor, signature_hash


# SECTION: experiment_and_outputs
def mathematical_checks():
    """Tiny synthetic arrays only: verify values and gradients, including second-order IRM."""
    cfg = default_config()
    generator = torch.Generator().manual_seed(11)
    base = torch.randn(4, 4, 7, generator=generator, dtype=torch.float64)
    labels = torch.tensor([[1, 2, 3, -100], [2, 1, -100, -100], [4, 2, 1, 5], [1, 3, -100, -100]])
    groups = torch.tensor([0, 0, 1, 1])
    weights = torch.tensor([0.8, 0.8, 1.2, 1.2], dtype=torch.float64)
    for method in ["ERM", "SD", "DRO", "IRM", "Fusion"]:
        logits = base.clone().requires_grad_(True)
        w = torch.ones((), dtype=torch.float64, requires_grad=True)
        token_ce = F.cross_entropy((w * logits).transpose(1, 2), labels, reduction="none")
        lengths = labels.ne(-100).sum(1)
        risks = token_ce.sum(1) / lengths
        erm = (risks * weights).mean()
        penalty = (((logits.square().mean(-1) * labels.ne(-100)).sum(1) / lengths) * weights).mean()
        sd = erm + cfg["sd_inner_lambda"] * penalty
        group_risks = torch.stack([risks[groups == g].mean() for g in [0, 1]])
        dro = group_risks.max()
        gradients = [torch.autograd.grad(r, w, create_graph=True, retain_graph=True)[0] for r in group_risks]
        irm_penalty = torch.stack(gradients).square().sum()
        pairs = []
        for gid in (0, 1):
            single = [torch.autograd.grad(r, w, create_graph=True, retain_graph=True)[0] for r in risks[groups == gid]]
            pairs.append(single[0]*single[1])
        irm = group_risks.mean() + cfg["irm_penalty_weight"]*torch.stack(pairs).mean()
        references = {"ERM": erm, "SD": sd, "DRO": dro, "IRM": irm,
                      "Fusion": cfg["lambda_e"]*erm + cfg["lambda_s"]*sd
                                + cfg["lambda_d"]*dro + cfg["lambda_i"]*irm_penalty}
        expected = references[method]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        actual_logits = base.clone().requires_grad_(True)
        actual, _ = objective(actual_logits, labels, groups, method, cfg, weights)
        actual_gradient = torch.autograd.grad(actual, actual_logits)[0]
        torch.testing.assert_close(actual, expected.detach(), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(actual_gradient[labels == -100], torch.zeros_like(actual_gradient[labels == -100]))
    padded_logits = torch.cat([base, torch.full((4, 2, 7), 999.0, dtype=torch.float64)], dim=1)
    padded_labels = torch.cat([labels, torch.full((4, 2), -100)], dim=1)
    for method in ["ERM", "SD", "DRO", "IRM", "Fusion"]:
        torch.testing.assert_close(objective(base, labels, groups, method, cfg)[0],
                                   objective(padded_logits, padded_labels, groups, method, cfg)[0])
    print("PASS: all five objectives and gradients match reference autograd; padding contributes zero.")


def validate_model_pipeline(processor, train_rows, test_rows, cfg):
    """Validate teacher forcing, gradients and decoding before training; exclude probe WER."""
    model, _ = load_backbone(cfg, trainable=True)
    try:
        collator = WhisperCollator(processor.tokenizer.pad_token_id)
        batch = move_batch(collator([FeatureDataset(train_rows)[i] for i in range(min(2, len(train_rows)))]), cfg)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype(cfg)):
            from transformers.models.whisper.modeling_whisper import shift_tokens_right
            torch.testing.assert_close(
                decoder_input_ids(batch["labels"], model.config.decoder_start_token_id, model.config.pad_token_id),
                shift_tokens_right(batch["labels"], model.config.pad_token_id, model.config.decoder_start_token_id))
            shifted_logits = forward_logits(model, batch)
            hf = model(input_features=batch["input_features"], attention_mask=batch["attention_mask"],
                       labels=batch["labels"], use_cache=False)
            torch.testing.assert_close(shifted_logits, hf.logits, atol=1e-3, rtol=1e-3)
            reference_cfg = {**cfg, "erm_reduction": "token_mean"}
            # Compare loss reductions on the SAME logits. Separate BF16 model
            # forwards may differ slightly even when the decoder shift is exact.
            token_mean = loss_components(hf.logits, batch["labels"], batch["groups"], reference_cfg)["ERM"]
            reference = F.cross_entropy(hf.logits.float().transpose(1, 2), batch["labels"])
            torch.testing.assert_close(token_mean.float(), reference, atol=1e-5, rtol=1e-5)
            # BF16 model loss can use a different reduction path. Require numerical
            # agreement within BF16 tolerance, while checking our FP32 arithmetic
            # strictly against the explicit reference on identical logits above.
            torch.testing.assert_close(token_mean.float(), hf.loss.float(), atol=1e-3, rtol=1e-3)
            ce_reference_difference = float((token_mean-reference).abs())
            hf_mixed_precision_difference = float((token_mean-hf.loss).abs())
        del shifted_logits, hf, token_mean, reference
        model.train()
        with torch.autocast("cuda", dtype=amp_dtype(cfg)):
            logits = forward_logits(model, batch)
            loss, _ = objective(logits, batch["labels"], batch["groups"], "Fusion", cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError("Model preflight objective is nonfinite")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError("Model preflight produced missing/nonfinite adapter gradients")
        model.zero_grad(set_to_none=True)
        del logits, loss, batch, grads
        long_rows = [r for r in test_rows if r["duration"] > 30]
        selected = test_rows[:2] + ([max(long_rows, key=lambda r: r["duration"])] if long_rows else [])
        smoke = evaluate(model, processor, selected, cfg, {r["group"] for r in selected}, "model pipeline check")
        atomic_json(Path(cfg["output_dir"]) / "model_preflight.json", {
            "status": "passed", "utterances": len(selected),
            "native_long_form_checked": any(r["duration"] > 30 for r in selected),
            "longest_checked_audio_seconds": max(r["duration"] for r in selected),
            "fp32_ce_reference_difference": ce_reference_difference,
            "hf_mixed_precision_ce_difference": hf_mixed_precision_difference,
            "inference_seconds": smoke["inference_seconds_this_session"],
        })
        print("PASS: model forward/backward, PEFT, batched decoding and long-form decoding.")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        seed_everything(cfg["seed"])


def initialize_experiment(cfg):
    from importlib.metadata import version
    from huggingface_hub import HfApi
    from transformers import WhisperProcessor
    configure_device(cfg)
    api = HfApi()
    # Preserve pinned commits across restarts, instead of resolving a changed 'main'.
    config_path = Path(cfg["output_dir"]) / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ["model_revision", "dataset_revision"]:
            if cfg[key] == "main":
                cfg[key] = previous[key]
        if cfg["eval_batch_size"] == "auto":
            cfg["eval_batch_size"] = previous["eval_batch_size"]
    cfg["model_revision"] = api.model_info(cfg["model_name"], revision=cfg["model_revision"]).sha
    cfg["dataset_revision"] = api.dataset_info(cfg["dataset_name"], revision=cfg["dataset_revision"]).sha
    required_disk = 5 if cached_features_ready(cfg) else 30
    free_gib = shutil.disk_usage(Path(cfg["cache_dir"])).free / 2**30
    if free_gib < required_disk:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free; this setup requires {required_disk} GiB.")
    print(f"Disk: {free_gib:.1f} GiB free; required headroom {required_disk} GiB.")
    cfg["package_versions"] = {name: version(name) for name in [
        "torch", "transformers", "peft", "datasets", "accelerate", "librosa", "soundfile", "jiwer", "numpy"]}
    processor = WhisperProcessor.from_pretrained(cfg["model_name"], revision=cfg["model_revision"],
                                                language=cfg["language"], task=cfg["task"])
    train, holdout, test = prepare_data(processor, cfg)
    if cfg["eval_batch_size"] == "auto":
        model, _ = load_backbone(cfg)
        try:
            cfg["eval_batch_size"] = tune_inference_batch_size(model, processor, test, cfg)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if config_path.exists():
        if json.loads(config_path.read_text()) != cfg:
            raise ValueError("Existing experiment configuration differs. Use a new output directory.")
    atomic_json(config_path, cfg)
    atomic_json(Path(cfg["output_dir"]) / "run_status.json", {"status": "in_progress"})
    return processor, train, holdout, test


def result_display_label(method, metrics):
    role = metrics.get("predictor_role")
    if role == "best_trained_checkpoint":
        return "IRM best trained checkpoint"
    status = metrics.get("selection", {}).get("status")
    if status == "pretrained_retained":
        return f"{method} selected pretrained fallback"
    return f"{method} selected trained checkpoint" if status == "finetuned_selected" else method


def write_results(results, cfg):
    output = Path(cfg["output_dir"])
    atomic_json(output / "measured_results.json", results)
    if "ERM" in results:
        atomic_json(output / "erm_goal_assessment.json", erm_goal_assessment(results))
    with (output / "results.csv").open("w", newline="") as stream:
        fields = ["method", "macro_wer", "minmax_gap", "groups", "paper_macro_wer", "paper_minmax_gap",
                  "macro_difference_from_paper", "gap_difference_from_paper",
                  "selected_epoch", "selection_status", "macro_delta_vs_baseline", "gap_delta_vs_baseline",
                  "both_metrics_below_baseline", "result_label", "predictor_role", "prediction_file"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, metrics in results.items():
            family = "IRM" if method == "IRM_trained" else method
            paper_macro, paper_gap = PAPER_SMALL[family]
            baseline = results.get("w/o FT")
            macro_delta = metrics["macro_wer"] - baseline["macro_wer"] if baseline else None
            gap_delta = metrics["minmax_gap"] - baseline["minmax_gap"] if baseline else None
            writer.writerow({"method": method, **{k: metrics[k] for k in ["macro_wer", "minmax_gap", "groups"]},
                             "paper_macro_wer": paper_macro, "paper_minmax_gap": paper_gap,
                             "macro_difference_from_paper": metrics["macro_wer"] - paper_macro,
                             "gap_difference_from_paper": metrics["minmax_gap"] - paper_gap,
                             "selected_epoch": metrics.get("selection", {}).get("selected_epoch"),
                             "selection_status": metrics.get("selection", {}).get("status", "pretrained_baseline"),
                             "macro_delta_vs_baseline": macro_delta, "gap_delta_vs_baseline": gap_delta,
                             "both_metrics_below_baseline": bool(
                                 baseline and macro_delta < 0 and gap_delta < 0 and
                                 metrics.get("selection", {}).get("status") != "pretrained_retained"),
                             "result_label": result_display_label(method, metrics),
                             "predictor_role": metrics.get("predictor_role", "pretrained_baseline" if method == "w/o FT" else "selected_checkpoint"),
                             "prediction_file": metrics.get("prediction_file")})
    with (output / "per_accent_wer.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["accent", *[result_display_label(method, metrics) for method, metrics in results.items()]])
        for group, accent in enumerate(PAPER_GROUPS):
            writer.writerow([accent, *[metrics["per_group_wer"][group] if group in metrics["per_group_wer"] else
                                       metrics["per_group_wer"][str(group)] for metrics in results.values()]])
    atomic_json(output / "result_labels.json", {method: {
        "result_label": result_display_label(method, metrics),
        "selection": metrics.get("selection"), "predictor_role": metrics.get("predictor_role"),
        "prediction_file": metrics.get("prediction_file")} for method, metrics in results.items()})


def run_baseline(test_rows, cfg, results):
    model, processor = load_backbone(cfg)
    try:
        results["w/o FT"] = evaluate(model, processor, test_rows, cfg, range(26), "w_o_FT",
                                      cache_signature=cfg["model_revision"])
        results["w/o FT"]["predictor_role"] = "pretrained_baseline"
        results["w/o FT"]["prediction_file"] = "predictions/w_o_FT.jsonl"
        write_results(results, cfg)
    finally:
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    return results["w/o FT"]


def run_method(strategy, train_rows, holdout_rows, test_rows, cfg, results):
    model, processor, signature = train_strategy(strategy, train_rows, holdout_rows, cfg)
    try:
        results[strategy] = evaluate(model, processor, test_rows, cfg, range(26), strategy,
                                     cache_signature=signature)
        results[strategy]["selection"] = json.loads(
            (Path(cfg["output_dir"]) / strategy / "selection.json").read_text())
        selection = results[strategy]["selection"]
        results[strategy]["predictor_role"] = ("pretrained_fallback" if selection["selected_epoch"] == 0
                                               else "selected_trained_checkpoint")
        results[strategy]["prediction_file"] = f"predictions/{strategy}.jsonl"
        if strategy == "IRM":
            trained_epoch = selection["best_trained_epoch"]
            if trained_epoch == selection["selected_epoch"]:
                trained_result = copy.deepcopy(results[strategy])
                trained_result["evaluation_reused_same_checkpoint"] = True
            else:
                # Select the best trained checkpoint on holdout only, even when it
                # loses to the pretrained control. Evaluate it separately for reporting.
                del model, processor
                gc.collect()
                torch.cuda.empty_cache()
                model, processor = load_backbone(cfg, trainable=True)
                path = Path(cfg["output_dir"]) / "IRM/best_trained_adapter.pt"
                if hashlib.sha256(path.read_bytes()).hexdigest() != selection["best_trained_sha256"]:
                    raise ValueError("Best trained adapter changed before test evaluation")
                restore_adapter(model, torch.load(path, map_location="cpu", weights_only=True))
                model = model.merge_and_unload()
                model.config.use_cache = True
                trained_result = evaluate(model, processor, test_rows, cfg, range(26), "IRM_best_trained",
                                          cache_signature=selection["best_trained_sha256"])
                trained_result["prediction_file"] = "predictions/IRM_best_trained.jsonl"
                trained_result["evaluation_reused_same_checkpoint"] = False
            trained_result["predictor_role"] = "best_trained_checkpoint"
            trained_result["selection"] = {"selected_epoch": trained_epoch, "status": "trained_checkpoint_reported",
                "selection_metric": "holdout_macro_wer_among_trained_epochs", "test_used_for_selection": False,
                "checkpoint_updates": selection["best_trained_updates"]}
            results["IRM_trained"] = trained_result
            for key in ("IRM", "IRM_trained"):
                metadata_path = Path(cfg["output_dir"]) / results[key]["prediction_file"]
                metadata_path = metadata_path.with_suffix(".metadata.json")
                metadata = json.loads(metadata_path.read_text())
                metadata.setdefault("reported_results", {})[key] = {
                    "predictor_role": results[key]["predictor_role"],
                    "checkpoint_epoch": results[key]["selection"]["selected_epoch"]}
                atomic_json(metadata_path, metadata)
        write_results(results, cfg)
    finally:
        # A failed reload after freeing the selected predictor must not mask its
        # original exception with an UnboundLocalError in cleanup.
        if "model" in locals():
            del model
        if "processor" in locals():
            del processor
        gc.collect()
        torch.cuda.empty_cache()
    return results[strategy]


def write_accent_analysis(results, cfg, train_rows, test_rows):
    """Describe measured accent coverage and word lengths for the selected methods."""
    import matplotlib.pyplot as plt
    grouped = defaultdict(list)
    for row in test_rows:
        grouped[row["group"]].append(row)
    training_counts = Counter(r["group"] for r in train_rows)
    records = []
    for group, accent in enumerate(PAPER_GROUPS):
        rows = grouped[group]
        words = [word for row in rows for word in normalize_text(row["reference"]).split()]
        record = {"accent": accent, "training_utterances": training_counts[group],
                  "test_utterances": len(rows), "test_speakers": len({r["speaker"] for r in rows}),
                  "test_hours": sum(r["duration"] for r in rows)/3600,
                  "reference_words": len(words),
                  "mean_word_length": float(np.mean([len(w) for w in words]))}
        for method, metrics in results.items():
            wer = metrics["per_group_wer"]
            record[method + "_wer"] = wer[group] if group in wer else wer[str(group)]
        records.append(record)
    output = Path(cfg["output_dir"])
    with (output / "accent_characteristics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lengths = np.array([r["mean_word_length"] for r in records])
    correlations = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in results:
        values = np.array([r[method + "_wer"] for r in records])
        correlations[method] = float(np.corrcoef(lengths, values)[0, 1]) if lengths.std() and values.std() else None
        ax.scatter(lengths, values, label=method, alpha=0.75)
    ax.set(xlabel="Mean reference word length (characters)", ylabel="Measured group WER (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "accent_properties.png", dpi=160)
    plt.show()
    plt.close(fig)
    atomic_json(output / "accent_analysis.json", {
        "word_length_wer_pearson": correlations,
        "interpretation": "Descriptive correlations on the complete 26-group test; not a causal result. "
                          "The paper gives no reproducible typological-distance matrix, so none is fabricated.",
    })


def finalize_experiment(results, cfg, train_rows, test_rows):
    import matplotlib.pyplot as plt
    expected = {"w/o FT", *cfg["strategies"]}
    if "IRM" in expected:
        expected.add("IRM_trained")
    if set(results) != expected or any(r["groups"] != 26 for r in results.values()):
        raise ValueError("Full experiment is incomplete; no success marker will be written.")
    write_results(results, cfg)
    write_accent_analysis(results, cfg, train_rows, test_rows)
    output = Path(cfg["output_dir"])
    angles = np.linspace(0, 2*np.pi, len(PAPER_GROUPS), endpoint=False)
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})
    for method in results:
        if method in results:
            values = [results[method]["per_group_wer"].get(g, results[method]["per_group_wer"].get(str(g))) for g in range(26)]
            ax.plot(np.r_[angles, angles[0]], values + values[:1], label=method)
    ax.set_xticks(angles, PAPER_GROUPS, fontsize=8)
    ax.set_title("Measured Whisper-small WER by accent (%)", pad=35)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(output / "accent_wer.png", dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for method in cfg["strategies"]:
        history = json.loads((output / method / "history.json").read_text())
        updates = [r for r in history if "loss" in r]
        validation = [r for r in history if "holdout_macro_wer" in r]
        axes[0].plot([r["update"] for r in updates], [r["loss"] for r in updates], label=method)
        axes[1].plot([r["epoch"] for r in validation], [r["holdout_macro_wer"] for r in validation], label=method)
    axes[0].set(xlabel="Optimizer update", ylabel="Method-specific training objective")
    axes[1].set(xlabel="Epoch", ylabel="Holdout macro-WER (%)")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.show()
    plt.close(fig)
    # Completion means every method decoded the full 26-group test, not that it matched the paper.
    atomic_json(output / "run_status.json", {
        "status": "complete", "interpretation": "Measured implementation of the paper's objectives; "
        "adapter, SD normalization, scalar IRM, training split and long-training exclusions are assumptions.",
        "paper_small_fusion_target": {"macro_wer": 30.3, "minmax_gap": 45.1},
        "fusion_measured": ({k: results["Fusion"][k] for k in ["macro_wer", "minmax_gap"]}
                            if "Fusion" in results else None),
        "erm_goal": erm_goal_assessment(results) if "ERM" in results else None,
    })
    print((output / "results.csv").read_text())
    if "ERM" in results:
        print("ERM improvement goal:", erm_goal_assessment(results)["status"])
    print("Complete. Outputs:", output.resolve())
