---
name: minimax-h3-lora-colab
description: Help curate, caption, and pair image datasets, then train a MiniMax H3 Ref2VA LoRA with the bundled Google Colab workflow. Use for H3 character or visual-style LoRA training, not Qwen Image training or ordinary H3 video generation.
---

# MiniMax H3 Ref2VA LoRA training

This skill trains an LoRA for the MiniMax H3 Ref2VA transformer with Ostris AI Toolkit on Google Colab. The H3 pipeline uses Qwen3-VL for image understanding; Qwen Image is a separate image-generation model and its LoRA weights are not an H3 adapter.

## Workflow

1. Help the user define what the LoRA should learn (one identity or one visual style), choose a unique trigger token, and identify which attributes should stay consistent versus vary. Keep unrelated people or concepts in separate datasets.
2. Inventory the supplied local photos and inspect them with available image-viewing tools. Note each basename, image quality, visible angle/pose, outfit, lighting, background, and obvious duplicates. Recommend excluding corrupt, very blurry, near-duplicate, or off-concept images; explain exclusions and let the user choose when quality is borderline. Put the selected target and reference photos in local staging folders by copying them; never alter or delete the originals. Do not upload images to Colab during dataset preparation.
3. Create a reviewable `dataset-plan.json` with one accurate caption for every target photo. Captions should describe visible content and useful variation while associating the intended identity/style with the trigger. Do not invent unseen details. When the user provides a separate reference folder, choose a useful same-subject reference for each target and map it explicitly. Example:

   ```json
   {
     "captions": {
       "photo_01.jpg": "unique_token person, waist-up portrait wearing a blue jacket, outdoors in soft daylight",
       "photo_02.jpg": "unique_token person, side profile in a white shirt, indoors near a window"
     },
     "references": {
       "photo_01.jpg": "front_reference.png",
       "photo_02.jpg": "front_reference.png"
     }
   }
   ```

   `captions` must cover every target filename. `references` is optional; when present, it must cover every target and each value must name an image in the reference candidates folder. Do not pair a target with itself; prefer varied useful views when several references fit. Save this JSON outside the photo staging folders.

4. Prepare the local H3 Ref2VA dataset:

   ```bash
   python3 scripts/runner.py prepare \
     --input-dir /path/to/target_photos \
     --reference-dir /path/to/reference_photos \
     --dataset-plan /path/to/dataset-plan.json \
     --name character_name \
     --trigger "unique_token person" \
     --output-dir ./prepared_dataset
   ```

   The command creates `targets/`, same-basename `references/`, captions, and `dataset_manifest.json` recording every target/reference/caption pair. Without `--dataset-plan`, it preserves existing sidecar captions and cycles through references as before. Review the manifest and report image count, caption coverage, pair choices, and any remaining visual-quality concerns. The runner checks file structure; visual quality still requires image review.

5. Training is a separate remote action. Before running it, explain that it starts a Colab GPU session, uploads the prepared images, and downloads H3 model assets. Run only when the user requested training. Do not retry a failed remote run automatically; report the Colab error and check whether a partial checkpoint was saved to Drive.

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
