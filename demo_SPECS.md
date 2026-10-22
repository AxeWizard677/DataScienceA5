# Demo specs: working pipeline for the live demo

Goal: a short notebook `demo_submitted.ipynb` (repo root) that runs D1 -> D2 -> D3 on any folder of images, and that I fully
understand. `Restart & Run All` passes without training and without COCO.

## Fixed decisions
- Pipeline = the one of section 13 of `deliverable3.ipynb` (the one described in the handed-in PDF):
  D1 `models/final_pretrained.keras` (MobileNetV2, 128x128), D2 U-Net `models/d2_unet_multi_*.weights.h5`
  (architecture rebuilt as in deliverable3.ipynb, cell "build_unet_any_size"), D3 `models/d3_final_captioner.keras`
  + `d3_final_vocab.json` + `d3_final_config.json` (ConvNeXt-Large frozen, 81 regions, GRU + attention, beam 3, no repeated 2-gram).
- NOT used: `*_improved.ipynb`, `pipeline.ipynb` (improved D1/D2: category classifier, gate, tiling, 8-orientation TTA).
- Test images: `demo_images/` (5 photos, 1 sketch, 1 diagram). Evaluation images come in a folder: only `INPUT_FOLDER` changes.
- A non-photo stops after D1 (no caption). Each step must also be callable alone (jury may inject an image anywhere).
- Image formats: any size, grey/RGBA converted to RGB (imageio, as in deliverable3.ipynb).

## Steps (each ends with printed checks, then wait for my validation)
0. Git: `git fetch origin`, `git checkout -b maxime/demo origin/simon/improvements`. Check: `git log -1`, `ls models demo_images`.
1. Environment: use the existing venv of the repo if it works; otherwise follow README "Setup". Check: TF/Keras versions,
   GPU listed, ConvNeXtLarge ImageNet weights downloaded (internet needed once), `ls models`.
2. Test `pipeline.ipynb` unchanged (fallback): Restart & Run All. Check: time, GPU memory use, results table, any error.
   Do not edit it. Report what works and what does not.
3. `demo_submitted.ipynb`, cell group A: imports + load D1, D2, D3 + encoder. Check: input/output shapes of each model, parameter counts.
4. Group B: three functions `classify(img)`, `denoise(img)`, `caption(img)`, each tested alone on one `demo_images/` image.
5. Group C: `run_pipeline(items)` chaining them, showing for each image the input, the denoised image, the caption.
   Check: one photo and the sketch (must stop after D1).
6. Group D: last cell with a single `INPUT_FOLDER`; results table saved to `reports/demo_results.csv`.
7. Final: Restart & Run All; time per image; GPU memory. Compare with the fallback `pipeline.ipynb` results on `demo_images/`.

## Pitfalls
- ConvNeXt-Large is run on CPU in the notebooks (2 GB GPU of a teammate). I have 8 GB: do NOT change the device unless I ask;
  if it is too slow, measure first and propose.
- D1 and D2 inputs are floats/pixels exactly as in deliverable3.ipynb section 13 (D1: pixels 0-255, 128x128; D2 and D3: see the
  notebook). Copy the preprocessing, do not reinvent it.
- Beam search code (`decode`) is defined in section 11.3 of deliverable3.ipynb. Reuse it, explain it, do not rewrite it.
- Same image, same caption: no random sampling at inference.

## If time runs short
Keep `pipeline.ipynb` as the demo. Never cut: a tested demo that runs from a fresh kernel.