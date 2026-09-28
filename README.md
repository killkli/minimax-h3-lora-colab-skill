# MiniMax H3 Ref2VA LoRA Colab Skill

An installable Codex skill for preparing paired image data and training a **MiniMax H3 Ref2VA LoRA** on Google Colab with Ostris AI Toolkit. The runner packages the dataset, runs the notebook, saves checkpoints to Google Drive, and downloads the final `.safetensors` file locally.

> This is an H3 training workflow. Qwen3-VL is used inside the H3 image-understanding pipeline; Qwen Image is a separate model and its LoRA weights are not compatible with the H3 transformer.

## Install

Requires Python 3.10+ locally and the Google Colab CLI installed and authenticated before using remote commands. To install the skill into the default Codex skills directory:

```bash
git clone https://github.com/killkli/minimax-h3-lora-colab-skill.git
cd minimax-h3-lora-colab-skill
./install.sh
```

By default, the installer uses `$CODEX_HOME/skills` when `CODEX_HOME` is set, or `~/.codex/skills` otherwise. To choose a different skills directory, use `./install.sh --dest /path/to/skills`. Existing installs are preserved unless `--force` is passed; replacement moves the prior copy to a timestamped backup.

Verify the installed command without starting a Colab job:

```bash
python3 ~/.codex/skills/minimax-h3-lora-colab/scripts/runner.py --help
```

If `CODEX_HOME` is set, substitute `$CODEX_HOME/skills` for `~/.codex/skills`.

## Prepare a dataset

Put target photos in one directory. Optional captions use the same basename and a `.txt` extension (`photo01.jpg` with `photo01.txt`). Use photos of the same person or subject.

```bash
python3 scripts/runner.py prepare \
  --input-dir /path/to/target_photos \
  --name studio_subject \
  --trigger "studio_subject person" \
  --output-dir ./prepared_dataset
```

The output layout is:

```text
prepared_dataset/
├── targets/       # target images and same-basename .txt captions
├── references/    # reference/control images with target-matched basenames
└── dataset_manifest.json  # source, target, reference, and caption for each sample
```

To select references from a separate folder, pass `--reference-dir /path/to/reference_photos`. Each reference must depict the same subject. Without a separate reference folder, provide at least two target photos; the runner cycles through them as references. If you curate a smaller set, keep the selected target and reference photos in local staging folders and leave the originals untouched.

### Agent-assisted dataset plan

The skill can inspect the user's local images, help curate target/reference pairs, write image-specific captions, and prepare a locally reviewable dataset. A plan JSON keeps those choices reproducible:

```json
{
  "captions": {
    "photo01.jpg": "studio_subject person, waist-up portrait in a blue jacket, outdoors in soft daylight",
    "photo02.jpg": "studio_subject person, side profile wearing a white shirt, indoors by a window"
  },
  "references": {
    "photo01.jpg": "front_reference.png",
    "photo02.jpg": "side_reference.png"
  }
}
```

`captions` must include every target image basename. `references` is optional; if present, it must map every target basename to an image basename in the candidate reference folder. Do not map a target to itself; prefer varied useful views where available. The Agent should describe visible details only, keep unrelated subjects in separate datasets, and review the generated `dataset_manifest.json` before any training upload.

```bash
python3 scripts/runner.py prepare \
  --input-dir /path/to/target_photos \
  --reference-dir /path/to/reference_photos \
  --dataset-plan /path/to/dataset-plan.json \
  --name studio_subject \
  --trigger "studio_subject person" \
  --output-dir ./prepared_dataset
```

When `--dataset-plan` is omitted, existing sidecar captions are used and missing captions get the simple default. Original photos are copied into the prepared dataset; preparation does not upload them to Colab.

## Train on Colab

Training consumes Colab GPU time and uploads the prepared images to the Colab runtime. The notebook downloads H3 model and training-adapter assets. Make sure the account can access Colab GPU hardware and has accepted the applicable asset licenses.

```bash
python3 scripts/runner.py train \
  --dataset-dir ./prepared_dataset \
  --name studio_subject \
  --trigger "studio_subject person" \
  --steps 1000 \
  --rank 16 \
  --alpha 16 \
  --gpu A100 \
  --output-dir ./outputs
```

Defaults are 1000 steps, rank 16, alpha 16, learning rate `1e-4`, and an A100 high-memory session. Add `--no-high-mem` only when the selected GPU allocation should omit Colab's high-memory request. The final adapter is saved in `./outputs/`; the notebook also syncs checkpoints, samples, config, and a run summary to `MyDrive/MiniMax_H3_LoRAs/<name>/<run-id>/`.

Other commands:

```bash
python3 scripts/runner.py usage --json   # read Colab compute balance
python3 scripts/runner.py test           # starts a remote GPU probe; not a training validation
python3 scripts/runner.py train --help
```

`test` provisions a Colab runtime and verifies only the remote probe, transfer, and Drive path. It does not validate that H3 training fits or completes.

## Hardware and validation status

The bundled notebook checks for 60 GiB of free disk and warns if VRAM is below 48 GiB. There is no official minimum-VRAM guarantee; a high-memory A100 is recommended, and actual fit depends on the Colab runtime. The upstream AI Toolkit Ref2VA example configuration is marked **unverified**, so this repo does not claim that every run will succeed. The notebook pins the inspected AI Toolkit source commit for reproducibility.

Local installation and CLI checks do not launch Colab training. No cloud training run is part of publishing this repository.

## References

- [MiniMax H3 model architecture](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/README.md#model-architecture)
- [Ostris AI Toolkit: MiniMax-H3 Ref2VA](https://ostris.com/docs/ai-toolkit/models/minimax-h3-ref2va)
- [Pinned AI Toolkit H3 implementation](https://github.com/ostris/ai-toolkit/blob/ecee894ed2b1f3716d9d7326693061ec1a3105bb/extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py)

## 中文說明

此 repo 訓練的是 MiniMax H3 Ref2VA LoRA。H3 流程使用 Qwen3-VL 處理影像訊號；Qwen Image 是另一個獨立模型，其 LoRA 權重不是 H3 adapter。Notebook 使用 Ostris AI Toolkit 的 `minimax_h3_ref2va` 架構，將目標照片和同名 reference/control 照片配對，輸出 H3 `.safetensors` LoRA。上游範例設定標示為未驗證；本機安裝確認不代表 Colab 雲端訓練已驗證。

Agent 也能協助整理本機訓練照片：檢視照片、建議保留或排除項目、為每張目標圖撰寫 caption，並選擇 reference 配對。透過 `--dataset-plan` 輸入逐圖 captions 和 reference 對照，runner 會產生 `dataset_manifest.json` 供檢查。這一步只在本機整理檔案；只有使用者另外要求訓練時，才會上傳資料到 Colab。
