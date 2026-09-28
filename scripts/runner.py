#!/usr/bin/env python3
"""Train MiniMax H3 Ref2VA LoRAs on Google Colab and archive them to Google Drive."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

SKILL_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK = SKILL_DIR / "assets" / "MiniMax_H3_LoRA_Train_Colab.ipynb"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
AUTH = os.environ.get("COLAB_AUTH", "oauth2")


class ColabError(RuntimeError):
    pass


def normalize_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-")[:64]
    if not name:
        raise ValueError("Name must contain at least one letter or number.")
    return name


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def clean_output(value: str) -> str:
    return ANSI_RE.sub("", value).replace("\r", "").strip()


def write_progress(path: Path | None, state: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def get_colab_bin() -> str:
    colab = shutil.which("colab")
    if not colab:
        home_local = Path.home() / ".local" / "bin" / "colab"
        if home_local.is_file():
            return str(home_local)
        raise FileNotFoundError("Google Colab CLI is not found on PATH or ~/.local/bin/colab.")
    return colab


def call_colab(
    arguments: list[str],
    *,
    label: str,
    timeout: float,
    on_line: Callable[[str], None] | None = None,
) -> str:
    command = [get_colab_bin(), f"--auth={AUTH}", *arguments]
    child = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        start_new_session=True,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def collect() -> None:
        assert child.stdout is not None
        for line in child.stdout:
            lines.put(line.rstrip("\n"))
        lines.put(None)

    reader = threading.Thread(target=collect, daemon=True)
    reader.start()
    output: list[str] = []
    eof = False
    started = time.monotonic()
    while not eof or child.poll() is None:
        if time.monotonic() - started > timeout and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except Exception:
                    pass
            reader.join(timeout=2)
            if child.stdout is not None:
                child.stdout.close()
            raise TimeoutError(f"{label} timed out after {timeout} seconds.")
        try:
            item = lines.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            eof = True
            continue
        line = clean_output(item)
        output.append(line)
        if on_line:
            on_line(line)
    returncode = child.wait()
    reader.join(timeout=2)
    if child.stdout is not None:
        child.stdout.close()
    if returncode != 0:
        tail = "\n".join(output[-15:])
        raise ColabError(f"{label} failed with exit code {returncode}:\n{tail}")
    return "\n".join(line for line in output if line)


def get_usage() -> dict[str, Any]:
    raw = call_colab(["usage"], label="colab usage", timeout=30)
    normalized = clean_output(raw)

    def extract(pattern: str, label: str) -> float:
        m = re.search(pattern, normalized, re.IGNORECASE | re.MULTILINE)
        if not m:
            raise ValueError(f"Could not parse {label} from colab usage.")
        return float(m.group(1).replace(",", ""))

    balance = extract(r"^Current balance:\s*([\d,]+(?:\.\d+)?)\s+compute units\s*$", "Current balance")
    rate = extract(r"^Usage rate:\s*([\d,]+(?:\.\d+)?)\s*/\s*hr\s*$", "Usage rate")
    assignments = extract(r"^Active assignments:\s*(\d+)\s*$", "Active assignments")
    return {
        "ok": True,
        "balance": balance,
        "rate_per_hour": rate,
        "active_assignments": int(assignments),
        "checked_at": now_iso(),
    }


def load_dataset_plan(
    plan_path: Path | None,
    target_names: set[str],
    reference_names: set[str],
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    if plan_path is None:
        return None, None

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or set(plan) - {"captions", "references"}:
        raise ValueError("Dataset plan must be a JSON object with only 'captions' and optional 'references'.")

    captions = plan.get("captions")
    if not isinstance(captions, dict):
        raise ValueError("Dataset plan must include a 'captions' object keyed by target image filename.")
    unknown_targets = set(captions) - target_names
    missing_captions = target_names - set(captions)
    if unknown_targets or missing_captions:
        problems = []
        if unknown_targets:
            problems.append(f"unknown target filenames: {', '.join(sorted(unknown_targets))}")
        if missing_captions:
            problems.append(f"missing captions for: {', '.join(sorted(missing_captions))}")
        raise ValueError("Dataset plan caption coverage error (" + "; ".join(problems) + ").")
    if any(not isinstance(text, str) or not text.strip() for text in captions.values()):
        raise ValueError("Every dataset plan caption must be a non-empty string.")
    normalized_captions = {name: text.strip() for name, text in captions.items()}

    references = plan.get("references")
    if references is None:
        return normalized_captions, None
    if not isinstance(references, dict):
        raise ValueError("Dataset plan 'references' must be an object keyed by target image filename.")
    unknown_targets = set(references) - target_names
    missing_references = target_names - set(references)
    invalid_references = [
        reference for reference in references.values()
        if not isinstance(reference, str) or reference not in reference_names
    ]
    if unknown_targets or missing_references or invalid_references:
        problems = []
        if unknown_targets:
            problems.append(f"unknown target filenames: {', '.join(sorted(unknown_targets))}")
        if missing_references:
            problems.append(f"missing reference choices for: {', '.join(sorted(missing_references))}")
        if invalid_references:
            problems.append(f"unknown reference filenames: {', '.join(sorted(map(str, invalid_references)))}")
        raise ValueError("Dataset plan reference coverage error (" + "; ".join(problems) + ").")
    return normalized_captions, references


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    name: str,
    trigger: str,
    reference_dir: Path | None = None,
    dataset_plan: Path | None = None,
) -> dict[str, Any]:
    name = normalize_name(name)
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    raw_images = sorted(p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_exts)
    if not raw_images:
        raise ValueError(f"No valid images found in {input_dir}. Supported: {valid_exts}")

    reference_images = raw_images
    if reference_dir is not None:
        reference_dir = reference_dir.resolve()
        reference_images = sorted(
            p for p in reference_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_exts
        )
        if not reference_images:
            raise ValueError(f"No valid reference images found in {reference_dir}. Supported: {valid_exts}")
    elif len(raw_images) < 2:
        raise ValueError("At least two input images are required when --reference-dir is not supplied.")

    captions_by_name, references_by_target = load_dataset_plan(
        dataset_plan,
        {image.name for image in raw_images},
        {image.name for image in reference_images},
    )
    if references_by_target is not None:
        reference_by_name = {image.name: image for image in reference_images}
        self_pairs = [
            target.name
            for target in raw_images
            if target.resolve() == reference_by_name[references_by_target[target.name]].resolve()
        ]
        if self_pairs:
            raise ValueError(
                "Dataset plan cannot use a target image as its own reference: "
                + ", ".join(self_pairs)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Prepared dataset output directory must be empty: {output_dir}")

    target_dir = output_dir / "targets"
    controls_dir = output_dir / "references"
    target_dir.mkdir()
    controls_dir.mkdir()
    trigger = trigger.strip() or f"{name} person"
    prepared: list[dict[str, str]] = []
    for idx, img in enumerate(raw_images):
        sample_number = idx + 1
        target_name = f"{name}_{sample_number:03d}{img.suffix.lower()}"
        target_img = target_dir / target_name
        if references_by_target is not None:
            selected_reference = references_by_target[img.name]
            reference = next(image for image in reference_images if image.name == selected_reference)
        else:
            reference = reference_images[(idx + 1) % len(reference_images)]
        if references_by_target is None and reference.resolve() == img.resolve() and len(reference_images) > 1:
            reference = reference_images[(idx + 2) % len(reference_images)]
        control_img = controls_dir / f"{target_img.stem}{reference.suffix.lower()}"
        shutil.copy2(img, target_img)
        shutil.copy2(reference, control_img)

        caption_file = target_img.with_suffix(".txt")
        existing_caption = img.with_suffix(".txt")
        if captions_by_name is not None:
            caption_text = captions_by_name[img.name]
        elif existing_caption.is_file() and existing_caption.stat().st_size > 0:
            caption_text = existing_caption.read_text(encoding="utf-8").strip()
        else:
            caption_text = ""
        if caption_text and trigger not in caption_text:
            caption_text = f"{trigger}, {caption_text}"
        if not caption_text:
            caption_text = f"A clear portrait photo of {trigger}, image {sample_number}."
        caption_file.write_text(caption_text, encoding="utf-8")
        prepared.append({
            "source_image": img.name,
            "target": target_img.name,
            "reference_source": reference.name,
            "reference": control_img.name,
            "caption": caption_text,
        })

    dataset_manifest = {
        "format": "minimax_h3_ref2va_dataset_v1",
        "name": name,
        "trigger": trigger,
        "count": len(prepared),
        "samples": prepared,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "ok": True,
        "character": name,
        "trigger": trigger,
        "count": len(prepared),
        "prepared_dir": str(output_dir),
        "targets_dir": str(target_dir),
        "references_dir": str(controls_dir),
        "manifest": str(manifest_path),
        "samples": prepared,
    }


def zip_dataset(dataset_dir: Path, zip_dest: Path) -> Path:
    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in dataset_dir.rglob("*"):
            if p.is_file() and not p.name.endswith(".zip"):
                arcname = p.relative_to(dataset_dir)
                z.write(p, arcname)
    return zip_dest


def run_train(
    *,
    dataset_dir: Path,
    name: str,
    trigger: str,
    steps: int,
    rank: int,
    alpha: int,
    lr: float,
    gpu: str,
    high_mem: bool,
    drive_path: str,
    output_dir: Path,
    session: str | None,
    keep_session: bool,
    timeout: float,
    progress_path: Path | None,
) -> dict[str, Any]:
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Colab training notebook not found at: {NOTEBOOK}")

    name = normalize_name(name)
    if steps < 1 or rank < 1 or alpha < 1 or lr <= 0:
        raise ValueError("steps, rank, alpha, and learning rate must all be positive.")
    drive_parts = Path(drive_path).parts
    if Path(drive_path).is_absolute() or ".." in drive_parts:
        raise ValueError("Google Drive path must be relative and must not contain '..'.")

    dataset_dir = dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {dataset_dir}")
    target_dir = dataset_dir / "targets"
    reference_dir = dataset_dir / "references"
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    targets = sorted(p for p in target_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_exts)
    references = sorted(p for p in reference_dir.glob("*") if p.is_file() and p.suffix.lower() in valid_exts)
    if not targets or not references:
        raise ValueError("Dataset must contain image files in both targets/ and references/.")
    reference_stems = {p.stem for p in references}
    for target in targets:
        if target.stem not in reference_stems:
            raise ValueError(f"No same-basename Ref2VA control image found for target: {target.name}")
        caption = target.with_suffix(".txt")
        if not caption.is_file() or not caption.read_text(encoding="utf-8").strip():
            raise ValueError(f"Target image is missing a non-empty caption: {caption}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    owns_session = session is None
    if session is None:
        session = f"h3-ref2va-lora-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:6]}"

    progress: dict[str, Any] = {
        "task": "h3_ref2va_lora_train",
        "architecture": "minimax_h3_ref2va",
        "status": "starting",
        "session": session,
        "run_id": run_id,
        "name": name,
        "trigger": trigger,
        "steps": steps,
        "rank": rank,
        "alpha": alpha,
        "drive_path": drive_path,
        "gpu": gpu,
        "log_tail": [],
        "updated_at": now_iso(),
    }
    write_progress(progress_path, progress)

    def log_line(line: str) -> None:
        if not line:
            return
        progress["log_tail"] = (progress["log_tail"] + [line])[-25:]
        progress["updated_at"] = now_iso()
        write_progress(progress_path, progress)

    work_dir = output_dir / f"train_work_{uuid.uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_zip = work_dir / "dataset.zip"
    zip_dataset(dataset_dir, dataset_zip)

    final_output = output_dir / f"{name}_minimax_h3_ref2va_{run_id}_rank{rank}_steps{steps}.safetensors"

    try:
        if owns_session:
            log_line(f"Starting Colab session {session} with GPU {gpu}...")
            start_args = ["new", "--session", session, "--gpu", gpu]
            if high_mem:
                start_args.append("--high-mem")
            call_colab(start_args, label="start Colab session", timeout=900, on_line=log_line)

        progress["status"] = "mounting_drive"
        write_progress(progress_path, progress)
        log_line("Mounting Google Drive to /content/drive...")
        try:
            call_colab(["drivemount", "--session", session, "/content/drive"], label="mount Google Drive", timeout=180, on_line=log_line)
            log_line("Google Drive successfully mounted via colab drivemount.")
        except Exception as drive_exc:
            log_line(f"Warning: colab drivemount reported: {drive_exc}. Will rely on notebook fallback.")

        progress["status"] = "uploading"
        write_progress(progress_path, progress)
        log_line("Uploading dataset archive...")
        call_colab(["upload", "--session", session, str(dataset_zip), "/content/dataset.zip"], label="upload dataset.zip", timeout=600, on_line=log_line)

        progress["status"] = "training"
        write_progress(progress_path, progress)
        env_vars = [
            f"LORA_NAME={name}",
            f"LORA_RUN_ID={run_id}",
            f"TRIGGER_WORD={trigger}",
            f"LORA_STEPS={steps}",
            f"LORA_RANK={rank}",
            f"LORA_ALPHA={alpha}",
            f"LORA_LR={lr}",
            f"DRIVE_SUBPATH={drive_path}",
        ]
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            dotenv_path = Path.home() / ".env"
            if dotenv_path.is_file():
                for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("HF_TOKEN="):
                        hf_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        if hf_token:
            env_vars.append(f"HF_TOKEN={hf_token}")

        exec_args = ["exec", "--session", session, "--timeout", str(timeout)]
        for env in env_vars:
            exec_args.extend(["--env", env])
        exec_args.extend(["--file", str(NOTEBOOK)])

        log_line("Executing LoRA training notebook on remote Colab kernel...")
        call_colab(exec_args, label="execute LoRA training", timeout=timeout + 60, on_line=log_line)

        progress["status"] = "downloading"
        write_progress(progress_path, progress)
        log_line("Downloading trained LoRA (.safetensors) to local storage...")
        call_colab(["download", "--session", session, "/content/lora_output.safetensors", str(final_output)], label="download LoRA", timeout=600, on_line=log_line)

        if not final_output.is_file() or final_output.stat().st_size == 0:
            raise FileNotFoundError(f"Trained LoRA output was not downloaded successfully: {final_output}")

        size_mb = round(final_output.stat().st_size / (1024**2), 2)
        log_line(f"LoRA training completed successfully! Output: {final_output.name} ({size_mb} MB)")

        progress.update({
            "status": "completed",
            "output_file": str(final_output),
            "output_size_mb": size_mb,
            "google_drive_folder": f"MyDrive/{drive_path}",
            "finished_at": now_iso(),
        })
        write_progress(progress_path, progress)
        return progress

    except Exception as exc:
        progress.update({"status": "failed", "error": str(exc), "finished_at": now_iso()})
        write_progress(progress_path, progress)
        raise

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        if owns_session and not keep_session:
            log_line("Stopping Colab session to prevent background compute usage...")
            try:
                call_colab(["stop", "--session", session], label="stop session", timeout=300, on_line=log_line)
            except Exception:
                pass


def run_test(
    *,
    gpu: str = "T4",
    drive_path: str = "MiniMax_H3_LoRAs/pipeline_smoke_test",
    output_dir: Path = Path("./test_output"),
    timeout: float = 300,
) -> dict[str, Any]:
    """Execute a lightweight end-to-end smoke test without full training."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = f"h3-test-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:6]}"

    print(f"[1/6] Initializing Colab test session '{session}' with GPU {gpu}...")
    call_colab(["new", "--session", session, "--gpu", gpu], label="start test session", timeout=600)

    try:
        print("[2/6] Testing Google Drive mounting...")
        drive_status = "mounted"
        try:
            call_colab(["drivemount", "--session", session, "/content/drive"], label="mount Google Drive", timeout=180)
            print("      Google Drive mounted successfully at /content/drive.")
        except Exception as e:
            drive_status = f"warning: {e}"
            print(f"      Drive mount notice: {e}")

        probe_code = f"""import json, os, shutil, sys, time, torch
from pathlib import Path

print("=== Colab Smoke Test Probe ===")
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
print("GPU Device:", gpu_name)

drive_ok = False
drive_target = Path("/content/drive/MyDrive/{drive_path}")
try:
    if Path("/content/drive/MyDrive").exists():
        drive_target.mkdir(parents=True, exist_ok=True)
        probe_info = {{
            "status": "success",
            "gpu": gpu_name,
            "session": "{session}",
            "tested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": "Colab upload/download and Google Drive access verified with a probe file; no H3 training was run."
        }}
        (drive_target / "probe_success.json").write_text(json.dumps(probe_info, indent=2), encoding="utf-8")
        drive_ok = True
        print("Successfully wrote probe_success.json to Google Drive!")
    else:
        print("Notice: /content/drive/MyDrive does not exist.")
except Exception as exc:
    print("Drive write error:", exc)

dummy_out = Path("/content/test_lora.safetensors")
try:
    from safetensors.torch import save_file
    save_file({{"lora_probe_weight": torch.ones(4, 4)}}, dummy_out)
except Exception:
    import struct
    dummy_out.write_bytes(b'{{"__metadata__":{{"probe":"ok"}},"test":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}' + b'\\x00'*16)

if drive_ok and drive_target.exists():
    shutil.copy2(dummy_out, drive_target / "test_lora.safetensors")
    print("Successfully synced test_lora.safetensors to Google Drive!")

print("=== Probe Execution Completed ===")
"""
        probe_file = output_dir / "probe.py"
        probe_file.write_text(probe_code, encoding="utf-8")

        print("[3/6] Uploading and executing remote verification probe...")
        call_colab(["upload", "--session", session, str(probe_file), "/content/probe.py"], label="upload probe.py", timeout=120)
        output_log = call_colab(["exec", "--session", session, "--file", str(probe_file), "--timeout", str(timeout)], label="run probe.py", timeout=timeout + 30)

        print("[4/6] Verifying file download pipeline...")
        local_safetensors = output_dir / "test_lora.safetensors"
        call_colab(["download", "--session", session, "/content/test_lora.safetensors", str(local_safetensors)], label="download test LoRA", timeout=180)

        assert local_safetensors.is_file() and local_safetensors.stat().st_size > 0, "Test LoRA was not downloaded."
        print(f"      Downloaded {local_safetensors.name} ({local_safetensors.stat().st_size} bytes).")

        return {
            "ok": True,
            "status": "passed",
            "session": session,
            "gpu": gpu,
            "drive_status": drive_status,
            "drive_target_path": f"MyDrive/{drive_path}",
            "downloaded_test_file": str(local_safetensors),
            "verified_at": now_iso(),
        }

    finally:
        print("[5/6] Stopping Colab test session to preserve compute balance...")
        try:
            call_colab(["stop", "--session", session], label="stop test session", timeout=180)
            print("      Test session stopped cleanly.")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniMax H3 Ref2VA LoRA trainer on Colab + Google Drive")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    usage_parser = subparsers.add_parser("usage", help="Check Colab compute units")
    usage_parser.add_argument("--json", action="store_true", help="Print JSON output")

    test_parser = subparsers.add_parser("test", help="Check Colab transfer and Drive access with a remote probe")
    test_parser.add_argument("--gpu", default="T4", help="GPU type for the remote probe (default: T4)")
    test_parser.add_argument("--drive-path", default="MiniMax_H3_LoRAs/pipeline_smoke_test", help="Google Drive test subfolder")
    test_parser.add_argument("--output-dir", type=Path, default=Path("./test_output"), help="Local test output folder")

    prep_parser = subparsers.add_parser("prepare", help="Prepare dataset images and captions")
    prep_parser.add_argument("--input-dir", required=True, type=Path, help="Input directory with photos")
    prep_parser.add_argument("--reference-dir", type=Path, default=None, help="Optional directory of same-character reference photos")
    prep_parser.add_argument("--dataset-plan", type=Path, default=None, help="Optional JSON with agent-authored captions and explicit target/reference pairs")
    prep_parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for training dataset")
    prep_parser.add_argument("--name", default="character", help="Character identifier name")
    prep_parser.add_argument("--trigger", default="", help="Trigger token in prompts (default: <name> person)")

    train_parser = subparsers.add_parser("train", help="Train a MiniMax H3 Ref2VA LoRA on Colab")
    train_parser.add_argument("--dataset-dir", required=True, type=Path, help="Prepared dataset containing targets/ and references/")
    train_parser.add_argument("--name", default="custom_lora", help="Unique LoRA model name")
    train_parser.add_argument("--trigger", default="", help="Trigger word (e.g. 'c_hero person')")
    train_parser.add_argument("--steps", type=int, default=1000, help="Training steps (default: 1000)")
    train_parser.add_argument("--rank", type=int, default=16, help="LoRA Rank (default: 16)")
    train_parser.add_argument("--alpha", type=int, default=16, help="LoRA Alpha (default: 16)")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    train_parser.add_argument("--gpu", default="A100", help="GPU type (default: A100)")
    train_parser.add_argument("--no-high-mem", action="store_true", help="Disable high-memory VM")
    train_parser.add_argument("--drive-path", default="", help="Subpath in Google Drive MyDrive (default: MiniMax_H3_LoRAs/<name>)")
    train_parser.add_argument("--output-dir", type=Path, default=Path("./outputs"), help="Local output directory")
    train_parser.add_argument("--session", default=None, help="Existing Colab session to reuse")
    train_parser.add_argument("--keep-session", action="store_true", help="Do not stop session after training")
    train_parser.add_argument("--timeout", type=float, default=10800, help="Timeout in seconds")
    train_parser.add_argument("--progress", type=Path, default=None, help="Path to progress.json")

    args = parser.parse_args()

    if args.subcommand == "usage":
        res = get_usage()
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(f"Colab Balance: {res['balance']} compute units | Rate: {res['rate_per_hour']}/hr | Active: {res['active_assignments']}")

    elif args.subcommand == "test":
        res = run_test(
            gpu=args.gpu,
            drive_path=args.drive_path,
            output_dir=args.output_dir,
        )
        print(json.dumps(res, indent=2))

    elif args.subcommand == "prepare":
        res = prepare_dataset(
            args.input_dir,
            args.output_dir,
            args.name,
            args.trigger,
            args.reference_dir,
            args.dataset_plan,
        )
        print(json.dumps(res, indent=2))

    elif args.subcommand == "train":
        drive_path = args.drive_path or f"MiniMax_H3_LoRAs/{args.name}"
        trigger = args.trigger or f"{args.name} person"
        res = run_train(
            dataset_dir=args.dataset_dir,
            name=args.name,
            trigger=trigger,
            steps=args.steps,
            rank=args.rank,
            alpha=args.alpha,
            lr=args.lr,
            gpu=args.gpu,
            high_mem=not args.no_high_mem,
            drive_path=drive_path,
            output_dir=args.output_dir,
            session=args.session,
            keep_session=args.keep_session,
            timeout=args.timeout,
            progress_path=args.progress,
        )
        print(json.dumps(res, indent=2))




if __name__ == "__main__":
    main()
