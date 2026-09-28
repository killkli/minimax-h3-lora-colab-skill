---
name: minimax-h3-lora-colab
description: Prepare paired image data and train a MiniMax H3 Ref2VA LoRA with the bundled Google Colab workflow. Use for H3 character or visual-style LoRA training, not Qwen Image training or ordinary H3 video generation.
---

# MiniMax H3 Ref2VA LoRA training

This skill trains an LoRA for the MiniMax H3 Ref2VA transformer with Ostris AI Toolkit on Google Colab. The H3 pipeline uses Qwen3-VL for image understanding; Qwen Image is a separate image-generation model and its LoRA weights are not an H3 adapter.

## Workflow

1. Check the requested target concept and inspect the supplied images and any same-basename `.txt` captions. Keep each dataset to one identity or visual concept.
2. Prepare paired targets and reference controls:

   ```bash
   python3 scripts/runner.py prepare \
     --input-dir /path/to/target_photos \
     --name character_name \
     --trigger "unique_token person" \
     --output-dir ./prepared_dataset
   ```

   The command creates `targets/` and `references/`; it pairs each target with the next image cyclically. With `--reference-dir`, candidates come from that directory. The control image must be a useful view of the same subject and have the same basename as its target; do not mix identities. Existing captions are retained and prefixed with the trigger when needed. Missing captions receive a simple trigger caption that the user can refine.

3. Before training, explain that the command starts a Colab GPU session, uploads the dataset to Colab, and downloads H3 model assets. Proceed when the user has requested training. Do not retry a failed remote run automatically; report the Colab error and check whether a partial checkpoint was saved to Drive.
4. Train only after the user requests the remote run:

   ```bash
   python3 scripts/runner.py train \
     --dataset-dir ./prepared_dataset \
     --name character_name \
     --trigger "unique_token person" \
     --steps 1000 --rank 16 --alpha 16 \
     --gpu A100 \
     --output-dir ./outputs
   ```

   The notebook uses `arch: minimax_h3_ref2va`, one-frame image targets, paired reference controls, and disables audio training. It writes a `.safetensors` LoRA locally and archives checkpoints and run metadata under `MyDrive/MiniMax_H3_LoRAs/<name>/`.

## Limits

- Local installation and CLI help do not run training. `usage` checks Colab compute balance; `test` starts a remote GPU probe and verifies transfer/Drive access only, not H3 training.
- The upstream AI Toolkit Ref2VA example config is marked unverified. Treat this notebook configuration as an implementation to validate on the user's Colab runtime, not as a guaranteed successful recipe.
- There is no published minimum VRAM guarantee. The notebook requires 60 GiB free disk and warns below 48 GiB VRAM; a high-memory A100 is recommended, but availability and fit depend on the runtime.
- This training skill produces an H3 Ref2VA adapter. It does not make the separate `minimax-h3-colab` video runner load that adapter unless that inference workflow explicitly supports custom H3 LoRA weights.
- Respect the model, training-adapter, and dataset licenses. Keep private training images out of this repository.

See `README.md` for installation, prerequisites, and full CLI options.
