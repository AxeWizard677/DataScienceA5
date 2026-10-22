# Demo notebooks: what differs

Both notebooks chain D1 (photo or not), D2 (denoising) and D3 (caption) on a folder of images. Only `INPUT_FOLDER` changes in the last cell.
`demo_submitted.ipynb` is the pipeline of section 13 of `deliverable3.ipynb` (the handed-in PDF). `pipeline.ipynb` is the improved fallback.

| Step | `demo_submitted.ipynb` | `pipeline.ipynb` |
|---|---|---|
| D1 | `final_pretrained.keras`: MobileNetV2, 128x128 input, photo if P >= 0.5. | `d1_improved_final.keras`: MobileNetV2, 224x224 input, photo if P >= 0.45. Adds a 5-category classifier (97.6 % test accuracy). A rejected image is repaired once by D2 and shown to D1 again. |
| D2 | `d2_unet_multi_20261019_100436.weights.h5`: U-Net with 3 pooling levels, 426,979 parameters, applied to every photo, at 288x288. | `d2_final_unet.keras`: U-Net with 4 pooling levels, 2,755,827 parameters, applied at the original size (256 px tiles in the code). A gate (correction >= 0.0275 or sharpening >= 1.3) leaves clean photos unchanged; a restored photo is the average of 8 orientations. |
| D3 | `d3_final_captioner.keras`, `d3_final_vocab.json`, `d3_final_config.json`: ConvNeXt-Large (frozen, CPU), 81 regions, GRU + attention, beam 3, no repeated 2-gram. | Same three files and the same settings. The decoding code is written differently, but the captions were identical for 11 of 11 photos (same image given to both). |

Verified in the notebooks' code, in `models/*.json` and by loading the models (input shapes, pooling layers, parameter counts).
The D3 check ran `caption()` of each notebook on `demo_images/01-05` and the 6 photos of `demo_eval/`.

`demo_eval/`: 12 test-split images of `protocol/manifest.csv` (seed 42): 6 Photo, 2 Text, 2 Painting, 1 Sketch, 1 Schematics.
Both notebooks classify 11 of 12 correctly; both accept `painting_09186.jpg` (a painting) as a photo.
