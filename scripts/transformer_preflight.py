#!/usr/bin/env python3
"""SANDARBH Phase 4A transformer/tokenizer/hardware preflight."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_PARTITIONS = frozenset({"train", "dev_tune"})
EXPECTED_COLUMNS = ("preceding", "target", "following", "A1_Score", "A2_Score", "A3_Score")
TEXT_FIELDS = ("target", "preceding", "following", "context")
TRUNCATION_FIELDS = (
    "example_id", "partition", "max_length", "ordinary_final_length", "target_token_count",
    "retained_target_token_count", "target_retention",
)
PACKAGE_NAMES = (
    "torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "numpy", "psutil",
)


class PreflightError(Exception):
    """Blocking Phase 4A setup, input, or validation error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_digest(values: Iterable[str]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def safe_path(relative: str, label: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PreflightError(f"{label} path escapes the project root") from exc
    return path


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be a JSON object")
    return value


def read_csv(path: Path, label: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise PreflightError(f"{label} has a missing or duplicate header")
            return list(reader.fieldnames), list(reader)
    except PreflightError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise PreflightError(f"Cannot read {label}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def installed_versions() -> dict[str, str]:
    result = {name: importlib.metadata.version(name) for name in PACKAGE_NAMES}
    result["python"] = platform.python_version()
    return result


def load_config(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path, "transformer preflight configuration")
    required = {
        "preflight_schema_version", "input_csv", "partition_artifacts", "reference_hashes",
        "required_partition_fingerprint", "tokenization_partition_allowlist", "model", "diagnostics",
        "proposed_input_policy", "benchmark", "installation", "outputs",
    }
    missing = sorted(required - set(config))
    if missing:
        raise PreflightError(f"Configuration keys missing: {missing}")
    if set(config["tokenization_partition_allowlist"]) != ALLOWED_PARTITIONS:
        raise PreflightError("Tokenization allowlist must contain exactly train and dev_tune")
    model = config["model"]
    if model.get("repository") != "distilbert/distilroberta-base":
        raise PreflightError("Unexpected pretrained repository")
    revision = model.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise PreflightError("Model revision must be an immutable 40-character commit")
    if model.get("trust_remote_code") is not False or model.get("use_safetensors") is not True:
        raise PreflightError("Preflight requires trust_remote_code=false and safetensors=true")
    if config["diagnostics"].get("sequence_lengths") != [256, 512]:
        raise PreflightError("Diagnostics must cover lengths 256 and 512")
    return config


def verify_frozen_inputs(config: dict[str, Any]) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    paths = {"source": safe_path(config["input_csv"], "source")}
    paths.update({key: safe_path(value, f"partition artifact {key}") for key, value in config["partition_artifacts"].items()})
    for name, path in paths.items():
        if not path.is_file():
            raise PreflightError(f"Missing {name}: {path}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    refs = config["reference_hashes"]
    comparisons = {
        "source": hashes["source"] == refs["source_sha256"],
        "targets": hashes["targets"] == refs["targets_sha256"],
        "split_manifest": hashes["split_manifest"] == refs["split_manifest_sha256"],
    }
    if not all(comparisons.values()):
        raise PreflightError(f"Frozen input hash mismatch: {comparisons}")
    status = load_json(paths["run_status"], "partition run status")
    summary = load_json(paths["summary"], "partition summary")
    if status.get("status") != "COMPLETED" or summary.get("status") != "FROZEN":
        raise PreflightError("Phase 2B is not completed and frozen")
    if summary.get("provenance_fingerprint") != config["required_partition_fingerprint"]:
        raise PreflightError("Partition fingerprint mismatch")
    return paths, hashes, summary


def load_source_rows(path: Path, desired_ids: set[str]) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader)
            if tuple(header) != EXPECTED_COLUMNS:
                raise PreflightError("Source header differs from the audited schema")
            for source_row, fields in enumerate(reader, 1):
                if len(fields) != len(header):
                    raise PreflightError(f"Malformed source record at source_row={source_row}")
                example_id = f"row-{source_row:06d}-{record_digest(fields)[:12]}"
                if example_id in desired_ids:
                    raw = dict(zip(header, fields))
                    selected[example_id] = {field: collapse_whitespace(raw[field]) for field in ("preceding", "target", "following")}
    except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
        raise PreflightError(f"Cannot resolve source examples: {exc}") from exc
    missing = sorted(desired_ids - set(selected))
    if missing:
        raise PreflightError(f"Source is missing requested stable IDs: {missing[:5]}")
    return selected


def load_tokenization_partitions(requested: Iterable[str], paths: dict[str, Path]) -> dict[str, list[dict[str, Any]]]:
    requested_list = list(requested)
    forbidden = sorted(set(requested_list) - ALLOWED_PARTITIONS)
    if forbidden:
        raise PreflightError(f"Phase 4A tokenization access denied for {forbidden}; allowed: train, dev_tune")
    _, manifest = read_csv(paths["split_manifest"], "split manifest")
    by_partition = {name: [row for row in manifest if row.get("partition") == name] for name in requested_list}
    desired_ids = {row["example_id"] for rows in by_partition.values() for row in rows}
    source = load_source_rows(paths["source"], desired_ids)
    output: dict[str, list[dict[str, Any]]] = {}
    for partition, rows in by_partition.items():
        rows.sort(key=lambda row: int(row["source_row"]))
        ids = [row["example_id"] for row in rows]
        if not ids or len(ids) != len(set(ids)):
            raise PreflightError(f"Empty or duplicate manifest IDs in {partition}")
        output[partition] = [{"example_id": row["example_id"], "partition": partition, **source[row["example_id"]]} for row in rows]
    return output


def context_with_target_span(fields: dict[str, str]) -> tuple[str, tuple[int, int]]:
    pieces: list[str] = []
    target_span: tuple[int, int] | None = None
    cursor = 0
    for field in ("preceding", "target", "following"):
        value = collapse_whitespace(fields.get(field, ""))
        if not value:
            continue
        if pieces:
            pieces.append("\n")
            cursor += 1
        start = cursor
        pieces.append(value)
        cursor += len(value)
        if field == "target":
            target_span = (start, cursor)
    if target_span is None:
        raise PreflightError("Eligible example has an empty target")
    return "".join(pieces), target_span


def retention_from_offsets(offsets: Iterable[Iterable[int]], target_span: tuple[int, int]) -> int:
    start, end = target_span
    return sum(1 for pair in offsets if len(pair) == 2 and int(pair[1]) > start and int(pair[0]) < end and int(pair[1]) > int(pair[0]))


def percentile(values: list[int], percent: float) -> float:
    if not values:
        raise PreflightError("Cannot summarize an empty length collection")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def summarize_lengths(before: list[int], final: list[int], thresholds: Iterable[int]) -> dict[str, Any]:
    def stats(values: list[int]) -> dict[str, Any]:
        return {"median": percentile(values, 0.5), "p95": percentile(values, 0.95), "maximum": max(values)}
    return {
        "before_special_tokens": stats(before),
        "final_including_special_tokens": stats(final),
        "counts_exceeding_final_sequence_length": {str(limit): sum(value > limit for value in final) for limit in thresholds},
    }


def diagnose_example(tokenizer: Any, row: dict[str, Any], thresholds: Iterable[int]) -> tuple[dict[str, tuple[int, int]], list[dict[str, Any]]]:
    context, span = context_with_target_span(row)
    values = {field: row[field] for field in ("target", "preceding", "following")}
    values["context"] = context
    lengths: dict[str, tuple[int, int]] = {}
    for field, value in values.items():
        before = tokenizer(value, add_special_tokens=False, truncation=False)["input_ids"]
        final = tokenizer(value, add_special_tokens=True, truncation=False)["input_ids"]
        lengths[field] = (len(before), len(final))
    untruncated = tokenizer(context, add_special_tokens=True, truncation=False, return_offsets_mapping=True)
    total_target = retention_from_offsets(untruncated["offset_mapping"], span)
    affected: list[dict[str, Any]] = []
    for limit in thresholds:
        truncated = tokenizer(context, add_special_tokens=True, truncation=True, max_length=limit, return_offsets_mapping=True)
        retained = retention_from_offsets(truncated["offset_mapping"], span)
        status = "full" if retained == total_target else ("complete_removal" if retained == 0 else "partial_removal")
        if status != "full":
            affected.append({
                "example_id": row["example_id"], "partition": row["partition"], "max_length": limit,
                "ordinary_final_length": len(untruncated["input_ids"]), "target_token_count": total_target,
                "retained_target_token_count": retained, "target_retention": status,
            })
    return lengths, affected


def token_diagnostics(tokenizer: Any, partitions: dict[str, list[dict[str, Any]]], thresholds: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result: dict[str, Any] = {
        "description": "Actual fast-tokenizer lengths; source text and token sequences are omitted.",
        "tokens_before_special_tokens": True,
        "final_lengths_include_special_tokens": True,
        "ordinary_context_order": ["preceding", "target", "following"],
        "ordinary_context_separator": "newline between nonempty fields",
        "partitions_inspected": list(partitions),
        "partitions": {},
    }
    all_affected: list[dict[str, Any]] = []
    aggregate = {field: {"before": [], "final": []} for field in TEXT_FIELDS}
    aggregate_retention = {str(limit): {"partial_removal": 0, "complete_removal": 0} for limit in thresholds}
    for partition, rows in partitions.items():
        collections = {field: {"before": [], "final": []} for field in TEXT_FIELDS}
        retention = {str(limit): {"partial_removal": 0, "complete_removal": 0} for limit in thresholds}
        for row in rows:
            lengths, affected = diagnose_example(tokenizer, row, thresholds)
            for field, (before, final) in lengths.items():
                collections[field]["before"].append(before)
                collections[field]["final"].append(final)
                aggregate[field]["before"].append(before)
                aggregate[field]["final"].append(final)
            for item in affected:
                retention[str(item["max_length"])][item["target_retention"]] += 1
                aggregate_retention[str(item["max_length"])][item["target_retention"]] += 1
            all_affected.extend(affected)
        result["partitions"][partition] = {
            "example_count": len(rows),
            "lengths": {field: summarize_lengths(values["before"], values["final"], thresholds) for field, values in collections.items()},
            "ordinary_right_truncation_target_effect": retention,
        }
    result["combined"] = {
        "example_count": sum(len(rows) for rows in partitions.values()),
        "lengths": {field: summarize_lengths(values["before"], values["final"], thresholds) for field, values in aggregate.items()},
        "ordinary_right_truncation_target_effect": aggregate_retention,
    }
    all_affected.sort(key=lambda row: (row["partition"], row["example_id"], int(row["max_length"])))
    return result, all_affected


def _windows_registry_hardware() -> tuple[str | None, list[dict[str, Any]]]:
    if os.name != "nt":
        return None, []
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except OSError:
        cpu = None
    gpus: list[dict[str, Any]] = []
    try:
        import winreg
        root_path = r"SYSTEM\CurrentControlSet\Control\Video"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root_path) as root:
            for index in range(winreg.QueryInfoKey(root)[0]):
                child = winreg.EnumKey(root, index)
                try:
                    with winreg.OpenKey(root, child + r"\0000") as key:
                        name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
                        driver = str(winreg.QueryValueEx(key, "DriverVersion")[0])
                        record = {"name": name, "driver_version": driver}
                        try:
                            raw_memory = winreg.QueryValueEx(key, "HardwareInformation.MemorySize")[0]
                            record["adapter_memory_bytes"] = int.from_bytes(raw_memory, "little") if isinstance(raw_memory, bytes) else int(raw_memory)
                        except OSError:
                            record["adapter_memory_bytes"] = None
                        if record not in gpus:
                            gpus.append(record)
                except OSError:
                    continue
    except OSError:
        pass
    return cpu, gpus


def inspect_environment(torch_module: Any) -> dict[str, Any]:
    import psutil
    cpu, gpus = _windows_registry_hardware()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(PROJECT_ROOT))
    cuda_available = bool(torch_module.cuda.is_available())
    cuda_devices = []
    if cuda_available:
        for index in range(torch_module.cuda.device_count()):
            properties = torch_module.cuda.get_device_properties(index)
            cuda_devices.append({"index": index, "name": properties.name, "total_memory_bytes": properties.total_memory})
    try:
        nvidia = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10, check=False)
        nvidia_info = {"available": nvidia.returncode == 0, "output": nvidia.stdout.strip() if nvidia.returncode == 0 else ""}
    except (OSError, subprocess.TimeoutExpired):
        nvidia_info = {"available": False, "output": ""}
    return {
        "os": {"description": platform.platform(), "system": platform.system(), "release": platform.release(), "version": platform.version(), "architecture": platform.machine()},
        "python": {"executable": sys.executable, "version": platform.python_version(), "implementation": platform.python_implementation()},
        "cpu": {"model": cpu or platform.processor() or "unavailable", "logical_processor_count": psutil.cpu_count(logical=True)},
        "memory": {"total_bytes": int(memory.total), "available_bytes": int(memory.available)},
        "disk": {"path": str(PROJECT_ROOT), "total_bytes": int(disk.total), "available_bytes": int(disk.free)},
        "display_adapters": gpus,
        "nvidia_smi": nvidia_info,
        "pytorch": {"version": torch_module.__version__, "cuda_is_available": cuda_available, "cuda_version": torch_module.version.cuda, "cuda_devices": cuda_devices},
        "actual_device": "cuda:0" if cuda_available else "cpu",
        "dependency_versions": installed_versions(),
    }


def require_finite_tensor(torch_module: Any, tensor: Any, label: str) -> None:
    if not bool(torch_module.isfinite(tensor).all().item()):
        raise PreflightError(f"{label} contains nonfinite values")


def verify_model(tokenizer: Any, model: Any, torch_module: Any) -> dict[str, Any]:
    encoded = tokenizer(["Synthetic transformer readiness check.", "Second synthetic example."], padding=True, truncation=True, max_length=32, return_tensors="pt")
    vocab_size = int(tokenizer.vocab_size)
    ids = encoded["input_ids"]
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
        raise PreflightError("Tokenizer produced an invalid token ID")
    model.eval()
    with torch_module.no_grad():
        logits = model(**encoded).logits
    if tuple(logits.shape) != (2, 2):
        raise PreflightError(f"Unexpected binary-logit shape: {tuple(logits.shape)}")
    require_finite_tensor(torch_module, logits, "Model verification logits")
    return {
        "model_loaded": True,
        "tokenizer_fast": bool(tokenizer.is_fast),
        "tokenizer_vocabulary_size": vocab_size,
        "special_tokens": {key: value for key, value in tokenizer.special_tokens_map.items()},
        "binary_logits_shape": [2, 2],
        "logits_finite": True,
        "token_ids_valid": True,
    }


def benchmark_worker(snapshot: str, output: Path, batch_size: int, sequence_length: int, warmup: int, timed: int, seed: int) -> int:
    import psutil
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    text = "Synthetic neutral benchmark sentence. " * 80
    batch = tokenizer([text] * batch_size, padding="max_length", truncation=True, max_length=sequence_length, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    labels = torch.tensor([index % 2 for index in range(batch_size)], dtype=torch.long, device=device)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    timings: list[float] = []
    finite = True
    model.train()
    for step in range(warmup + timed):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        output_value = model(**batch, labels=labels)
        finite = finite and bool(torch.isfinite(output_value.loss).item()) and bool(torch.isfinite(output_value.logits).all().item())
        output_value.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step >= warmup:
            timings.append(elapsed)
    record = {
        "status": "COMPLETED", "device": str(device), "precision": "float32", "microbatch_size": batch_size,
        "sequence_length": sequence_length, "warmup_steps": warmup, "measured_steps": timed,
        "step_seconds": timings, "mean_seconds_per_step": sum(timings) / len(timings),
        "process_peak_rss_bytes_observed_at_step_boundaries": int(peak_rss), "outputs_finite": finite,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "includes_forward_backward_optimizer_and_gradient_reset": True,
    }
    write_json(output, record)
    return 0 if finite else 3


def run_benchmarks(snapshot: Path, config: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    settings = config["benchmark"]
    start = time.perf_counter()
    records: list[dict[str, Any]] = []
    for batch_size in settings["microbatch_sizes"]:
        remaining = settings["overall_timeout_seconds"] - (time.perf_counter() - start)
        if remaining <= 5:
            records.append({"status": "SKIPPED_OVERALL_TIMEOUT", "microbatch_size": batch_size, "sequence_length": settings["sequence_length"]})
            break
        timeout = min(float(settings["per_configuration_timeout_seconds"]), remaining)
        with tempfile.TemporaryDirectory(prefix="phase4a-benchmark-") as temp:
            result_path = Path(temp) / "result.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--benchmark-worker", "--snapshot", str(snapshot), "--worker-output", str(result_path), "--batch-size", str(batch_size), "--sequence-length", str(settings["sequence_length"]), "--warmup-steps", str(settings["warmup_steps"]), "--timed-steps", str(settings["timed_steps"]), "--seed", str(settings["seed"])]
            try:
                completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
                if completed.returncode == 0 and result_path.is_file():
                    record = load_json(result_path, "benchmark worker output")
                else:
                    lower = (completed.stderr + completed.stdout).lower()
                    record = {"status": "OOM" if "out of memory" in lower else "FAILED", "microbatch_size": batch_size, "sequence_length": settings["sequence_length"], "exit_code": completed.returncode, "error": (completed.stderr or completed.stdout)[-1000:]}
            except subprocess.TimeoutExpired:
                record = {"status": "TIMED_OUT", "microbatch_size": batch_size, "sequence_length": settings["sequence_length"], "timeout_seconds": timeout}
        records.append(record)
        if record.get("status") != "COMPLETED":
            break
    completed_records = [record for record in records if record.get("status") == "COMPLETED"]
    estimate = None
    if completed_records:
        preferred = completed_records[-1]
        steps_per_epoch = math.ceil(settings["training_examples"] / preferred["microbatch_size"])
        total_steps = steps_per_epoch * settings["epochs_for_estimate"]
        estimate = {
            "based_on_microbatch_size": preferred["microbatch_size"],
            "mean_seconds_per_step": preferred["mean_seconds_per_step"],
            "steps_per_epoch_ceiling": steps_per_epoch,
            "epochs": settings["epochs_for_estimate"],
            "total_optimizer_steps": total_steps,
            "estimated_seconds": total_steps * preferred["mean_seconds_per_step"],
            "calculation": "ceil(1595 / microbatch_size) * 3 * measured_mean_seconds_per_step",
            "qualification": "Linear extrapolation only; validation, checkpointing, variable lengths, data loading, and hardware load are excluded.",
        }
    return {
        "status": "COMPLETED" if completed_records else "BENCHMARK_UNAVAILABLE",
        "device": environment["actual_device"], "precision": settings["precision"],
        "available_memory_before_benchmark_bytes": environment["memory"]["available_bytes"],
        "configurations": records, "three_epoch_extrapolation": estimate,
        "overall_elapsed_seconds": time.perf_counter() - start,
        "synthetic_only": True, "updated_weights_and_optimizer_state_discarded": True,
        "sequence_length_512_speed_unmeasured": True,
    }


def model_snapshot(config: dict[str, Any]) -> tuple[Path, str]:
    from huggingface_hub import HfApi, snapshot_download
    model = config["model"]
    info = HfApi().model_info(model["repository"], revision=model["revision"])
    if info.sha != model["revision"]:
        raise PreflightError(f"Resolved model revision differs: {info.sha}")
    snapshot = snapshot_download(
        repo_id=model["repository"], revision=model["revision"],
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"],
    )
    path = Path(snapshot)
    if not (path / "model.safetensors").is_file():
        raise PreflightError("Safetensors model weight was not downloaded")
    if any(path.glob("*.bin")):
        raise PreflightError("Duplicate PyTorch .bin weights are present in the resolved snapshot")
    return path, info.sha


def proposed_policy_record(tokenizer: Any, config: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    policy = config["proposed_input_policy"]
    markers = policy["markers"]
    marker_lengths = {name: len(tokenizer(value, add_special_tokens=False)["input_ids"]) for name, value in markers.items()}
    special_count = len(tokenizer("", add_special_tokens=True, truncation=False)["input_ids"])
    max_length = int(policy["sequence_length"])
    target_cap = max_length - special_count - sum(marker_lengths.values())
    effects = diagnostics["combined"]["ordinary_right_truncation_target_effect"][str(max_length)]
    return {
        "status": "PROPOSED_NOT_APPROVED_OR_TRAINED",
        "recommended_max_length": max_length,
        "basis": "Actual tokenizer length and target-retention diagnostics plus measured hardware readiness; labels and model performance were not used.",
        "outer_special_token_count": special_count,
        "marker_token_counts": marker_lengths,
        "maximum_retained_target_tokens": target_cap,
        "ordering": ["outer_start", "preceding_marker", "preceding_tail", "target_marker", "target_prefix", "following_marker", "following_head", "outer_end"],
        "missing_context": "Keep the marker and use zero content tokens; redistribute unused content capacity deterministically.",
        "overlong_target": policy["overlong_target"],
        "context_allocation": policy["context_allocation"],
        "ordinary_right_truncation_affected_at_recommended_length": effects,
        "target_only_and_context_same_target_retention_rule": True,
        "hard_and_soft_context_inputs_identical": True,
    }


def markdown_report(manifest: dict[str, Any], environment: dict[str, Any], diagnostics: dict[str, Any], benchmark: dict[str, Any]) -> str:
    lines = [
        "# Phase 04A Transformer Preflight", "", f"**Status:** `{manifest['status']}`  ",
        f"**Generated (UTC):** {manifest['timestamp_utc']}  ", "",
        "Phase 4A loaded the fixed DistilRoBERTa tokenizer and binary classification model, tokenized only `train` and `dev_tune` for diagnostics, and ran a synthetic optimizer benchmark. It did not fine-tune on AUTALIC or generate real predictions.", "",
        "## Environment", "",
        f"- Device: `{environment['actual_device']}`; `torch.cuda.is_available()` = `{str(environment['pytorch']['cuda_is_available']).lower()}`.",
        f"- CPU: {environment['cpu']['model']} ({environment['cpu']['logical_processor_count']} logical processors).",
        f"- RAM: {environment['memory']['total_bytes']} bytes total; {environment['memory']['available_bytes']} bytes available at inspection.",
        f"- Model revision: `{manifest['model']['resolved_revision']}`.", "",
        "## Token lengths", "",
        "All values are actual tokenizer counts. Final lengths include special tokens.", "",
        "| Partition | Field | Median before/final | P95 before/final | Max before/final | >256 | >512 |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for partition, item in diagnostics["partitions"].items():
        for field, values in item["lengths"].items():
            before, final = values["before_special_tokens"], values["final_including_special_tokens"]
            exceed = values["counts_exceeding_final_sequence_length"]
            lines.append(f"| {partition} | {field} | {before['median']:.1f}/{final['median']:.1f} | {before['p95']:.1f}/{final['p95']:.1f} | {before['maximum']}/{final['maximum']} | {exceed['256']} | {exceed['512']} |")
    lines.extend(["", "## Ordinary right-truncation effect on target", "", "| Partition | Limit | Partial target removal | Complete target removal |", "|---|---:|---:|---:|"])
    for partition, item in diagnostics["partitions"].items():
        for limit, counts in item["ordinary_right_truncation_target_effect"].items():
            lines.append(f"| {partition} | {limit} | {counts['partial_removal']} | {counts['complete_removal']} |")
    lines.extend(["", "## Synthetic benchmark", ""])
    for record in benchmark["configurations"]:
        if record.get("status") == "COMPLETED":
            lines.append(f"- Batch {record['microbatch_size']}, length {record['sequence_length']}, FP32: {record['mean_seconds_per_step']:.3f} seconds/step across {record['measured_steps']} measured steps (`{record['device']}`).")
        else:
            lines.append(f"- Batch {record.get('microbatch_size')}: `{record.get('status')}`.")
    estimate = benchmark.get("three_epoch_extrapolation")
    if estimate:
        lines.append(f"- Three-epoch linear extrapolation: {estimate['estimated_seconds']:.1f} seconds = ceil(1595/{estimate['based_on_microbatch_size']}) × 3 × {estimate['mean_seconds_per_step']:.3f}. This excludes validation, checkpointing, variable lengths, data loading, and hardware contention.")
    policy = manifest["proposed_input_policy"]
    lines.extend(["", "## Recommendation", "", f"Use a proposed maximum length of **{policy['recommended_max_length']}** with explicit role markers, target-first budgeting, preceding-tail retention, and following-head retention. Overlong targets retain the declared prefix cap. E6 and E7 must have identical input IDs and masks. This is a proposed next-phase policy, not an approved or executed training setup.", "", "## Boundaries", "", "No real fine-tuning, real classification prediction, calibration, threshold selection, transformer checkpoint saving, `dev_calibration` tokenization, or `test` tokenization/evaluation occurred. The synthetic optimizer state and updated weights were discarded.", ""])
    return "\n".join(lines)


def text_free_schema(value: Any, forbidden_keys: frozenset[str] = frozenset({"text", "tokens", "input_ids", "decoded", "source_sentence"})) -> bool:
    if isinstance(value, dict):
        return not (set(value) & forbidden_keys) and all(text_free_schema(item, forbidden_keys) for item in value.values())
    if isinstance(value, list):
        return all(text_free_schema(item, forbidden_keys) for item in value)
    return True


def validate_outputs(config_path: Path, quiet: bool = False) -> dict[str, Any]:
    failures: list[str] = []
    checks: dict[str, bool] = {}
    try:
        config = load_config(config_path)
        paths, hashes, summary = verify_frozen_inputs(config)
        outputs = {name: safe_path(value, f"output {name}") for name, value in config["outputs"].items()}
        required = ("environment", "token_length_summary", "truncation_examples", "benchmark", "run_manifest", "run_status", "report")
        checks["all_outputs_exist"] = all(outputs[name].is_file() for name in required)
        if not checks["all_outputs_exist"]:
            raise PreflightError("One or more required outputs are missing")
        environment = load_json(outputs["environment"], "environment output")
        tokens = load_json(outputs["token_length_summary"], "token summary")
        benchmark = load_json(outputs["benchmark"], "benchmark output")
        manifest = load_json(outputs["run_manifest"], "run manifest")
        status = load_json(outputs["run_status"], "run status")
        fields, truncation = read_csv(outputs["truncation_examples"], "truncation examples")
        checks["status_completed"] = status.get("status") == "COMPLETED" and manifest.get("status") == "COMPLETED"
        checks["partition_fingerprint_matches"] = manifest.get("partition_fingerprint") == summary.get("provenance_fingerprint") == config["required_partition_fingerprint"]
        checks["input_hashes_match"] = manifest.get("input_hashes") == hashes
        checks["code_config_hashes_match"] = manifest.get("code_and_config_hashes") == {
            "config": sha256_file(config_path), "script": sha256_file(Path(__file__)),
            "requirements": sha256_file(PROJECT_ROOT / "requirements-transformers.txt"),
            "plan": sha256_file(PROJECT_ROOT / "reports" / "PHASE_04_TRANSFORMER_PLAN.md"),
        }
        checks["only_allowed_partitions_inspected"] = tokens.get("partitions_inspected") == ["train", "dev_tune"] and manifest.get("tokenization_partitions_accessed") == ["train", "dev_tune"]
        checks["example_counts_match"] = tokens.get("partitions", {}).get("train", {}).get("example_count") == 1595 and tokens.get("partitions", {}).get("dev_tune", {}).get("example_count") == 171
        checks["model_revision_matches"] = manifest.get("model", {}).get("resolved_revision") == config["model"]["revision"] == manifest.get("model", {}).get("requested_revision")
        checks["model_verification_passed"] = all(manifest.get("model_verification", {}).get(key) is True for key in ("model_loaded", "logits_finite", "token_ids_valid")) and manifest.get("model_verification", {}).get("binary_logits_shape") == [2, 2]
        checks["text_free_json_schemas"] = all(text_free_schema(value) for value in (environment, tokens, benchmark, manifest, status))
        checks["truncation_csv_schema_text_free"] = fields == list(TRUNCATION_FIELDS) and all(set(row) == set(TRUNCATION_FIELDS) for row in truncation)
        named = {key: outputs[key] for key in ("environment", "token_length_summary", "truncation_examples", "benchmark")}
        checks["output_hashes_match"] = all(manifest.get("output_hashes", {}).get(key) == sha256_file(path) for key, path in named.items())
        checks["no_real_training_or_predictions"] = manifest.get("real_fine_tuning_performed") is False and manifest.get("real_predictions_generated") is False
        checks["synthetic_state_discarded"] = benchmark.get("synthetic_only") is True and benchmark.get("updated_weights_and_optimizer_state_discarded") is True
        for name, passed in checks.items():
            if not passed:
                failures.append(f"Check failed: {name}")
    except PreflightError as exc:
        failures.append(str(exc))
    result = {"passed": not failures, "checks": checks, "failures": failures}
    if not quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_preflight(config_path: Path) -> int:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    config = load_config(config_path)
    paths, input_hashes, partition_summary = verify_frozen_inputs(config)
    outputs = {name: safe_path(value, f"output {name}") for name, value in config["outputs"].items()}
    timestamp = utc_now()
    write_json(outputs["run_status"], {"status": "RUNNING", "timestamp_utc": timestamp, "message": "Phase 4A preflight is running."})
    protected_before = {name: sha256_file(path) for name, path in paths.items()}
    try:
        environment = inspect_environment(torch)
        snapshot, resolved_revision = model_snapshot(config)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=True)
        if not tokenizer.is_fast:
            raise PreflightError("Fast tokenizer with offset mappings is required")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model, loading_info = AutoModelForSequenceClassification.from_pretrained(
                snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                num_labels=config["model"]["num_labels"], output_loading_info=True,
            )
        verification = verify_model(tokenizer, model, torch)
        verification["classification_head_initialization"] = {
            "expected": True,
            "missing_keys": sorted(loading_info.get("missing_keys", [])),
            "unexpected_keys": sorted(loading_info.get("unexpected_keys", [])),
            "warning_messages": [str(item.message) for item in caught],
            "interpretation": "A new binary classification head is expected and is not evidence of a failed base-model download.",
        }
        del model
        partitions = load_tokenization_partitions(["train", "dev_tune"], paths)
        diagnostics, truncation_rows = token_diagnostics(tokenizer, partitions, config["diagnostics"]["sequence_lengths"])
        diagnostics.update({
            "model_repository": config["model"]["repository"], "model_revision": resolved_revision,
            "normalization": config["diagnostics"]["normalization"], "case_preserved": True,
            "percentile_method": config["diagnostics"]["percentile_method"],
        })
        policy = proposed_policy_record(tokenizer, config, diagnostics)
        benchmark = run_benchmarks(snapshot, config, environment)
        code_hashes = {
            "config": sha256_file(config_path), "script": sha256_file(Path(__file__)),
            "requirements": sha256_file(PROJECT_ROOT / "requirements-transformers.txt"),
            "plan": sha256_file(PROJECT_ROOT / "reports" / "PHASE_04_TRANSFORMER_PLAN.md"),
        }
        with tempfile.TemporaryDirectory(prefix="phase4a-output-", dir=PROJECT_ROOT) as temp:
            staged = Path(temp)
            staged_paths = {name: staged / Path(path).name for name, path in outputs.items() if name not in {"run_status", "report"}}
            write_json(staged_paths["environment"], environment)
            write_json(staged_paths["token_length_summary"], diagnostics)
            write_csv(staged_paths["truncation_examples"], TRUNCATION_FIELDS, truncation_rows)
            write_json(staged_paths["benchmark"], benchmark)
            output_hashes = {name: sha256_file(staged_paths[name]) for name in ("environment", "token_length_summary", "truncation_examples", "benchmark")}
            manifest = {
                "status": "COMPLETED", "timestamp_utc": timestamp,
                "preflight_schema_version": config["preflight_schema_version"],
                "partition_fingerprint": partition_summary["provenance_fingerprint"],
                "input_hashes": input_hashes, "code_and_config_hashes": code_hashes,
                "output_hashes": output_hashes,
                "model": {"repository": config["model"]["repository"], "requested_revision": config["model"]["revision"], "resolved_revision": resolved_revision, "weight_format": "safetensors", "trust_remote_code": False},
                "model_verification": verification,
                "tokenization_partitions_accessed": ["train", "dev_tune"],
                "tokenization_counts": {name: len(rows) for name, rows in partitions.items()},
                "benchmark_status": benchmark["status"], "training_feasibility_recommendation": "FUNCTIONALLY_FEASIBLE" if benchmark["status"] == "COMPLETED" else "BENCHMARK_REVIEW_NEEDED",
                "proposed_input_policy": policy,
                "real_fine_tuning_performed": False, "real_predictions_generated": False,
                "not_performed": ["AUTALIC transformer fine-tuning", "dev_calibration tokenization", "test tokenization", "real classification prediction", "calibration", "threshold selection", "final evaluation"],
            }
            write_json(staged_paths["run_manifest"], manifest)
            if {name: sha256_file(path) for name, path in paths.items()} != protected_before:
                raise PreflightError("Source or frozen partition artifact changed during preflight")
            for name in ("environment", "token_length_summary", "truncation_examples", "benchmark", "run_manifest"):
                outputs[name].parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_paths[name], outputs[name])
        outputs["report"].parent.mkdir(parents=True, exist_ok=True)
        with outputs["report"].open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(markdown_report(manifest, environment, diagnostics, benchmark))
        write_json(outputs["run_status"], {"status": "COMPLETED", "timestamp_utc": timestamp, "message": "Phase 4A transformer preflight completed.", "benchmark_status": benchmark["status"], "training_feasibility_recommendation": manifest["training_feasibility_recommendation"]})
        validation = validate_outputs(config_path, quiet=True)
        if not validation["passed"]:
            raise PreflightError(f"Generated output validation failed: {validation['failures']}")
        print("Phase 4A transformer preflight completed: COMPLETED")
        print(f"Manifest: {outputs['run_manifest'].relative_to(PROJECT_ROOT)}")
        return 0
    except Exception as exc:
        write_json(outputs["run_status"], {"status": "SETUP_BLOCKER" if isinstance(exc, (OSError, ImportError)) else "FAILED", "timestamp_utc": timestamp, "message": str(exc)})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transformer_preflight.json")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--benchmark-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sequence-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--warmup-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--timed-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.benchmark_worker:
        return benchmark_worker(args.snapshot, Path(args.worker_output), args.batch_size, args.sequence_length, args.warmup_steps, args.timed_steps, args.seed)
    config_path = safe_path(args.config, "configuration")
    if args.validate_only:
        return 0 if validate_outputs(config_path)["passed"] else 1
    try:
        return run_preflight(config_path)
    except PreflightError as exc:
        print(f"Phase 4A preflight failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Phase 4A preflight failed unexpectedly: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
