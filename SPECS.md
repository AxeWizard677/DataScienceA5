# Leyenda project: specifications

## Context
Client: TouNum, a document digitization company with no ML specialist. We build a
prototype that automatically captions photographs. Two difficulties besides the
captioning itself: scans have variable quality (blurred, noisy), and the digitized
base mixes photos with other images (text, diagrams, drawings, paintings).

Workflow: load images -> binary classification (photo / not photo) -> denoising
-> captioning.

Stack: Python, TensorFlow/Keras, scikit-learn, Pandas, NumPy, Matplotlib, ImageIO.
Each deliverable is a Jupyter notebook: didactic, reproducible, with supporting
statistics. Deliverable 3 repeats the first two and presents the whole workflow.
Final defense: 20 minutes of presentation (workflow, results, recommendations), then
a demo on an unseen dataset, then questions. The demo checks that the pipeline steps
chain together. Students must be able to explain and justify their work critically.
Notebooks and the PDF report are written in English.

## Shared technical conventions
- Images: RGB, float32 in [0, 1] when fed to a D1/D2 model. D1/D2 work at 128x128.
  D3 works at the size required by the chosen backbone (299 for InceptionV3), see
  "Image size decision" in D3.
- Memory: some teammates have a 2 GB GPU. Small batches, tf.data, uint8 in memory
  for images, float16 on disk for cached features.
- Reproducibility: fixed seeds, pinned versions, relative paths only.
- Large data (data/, cache/, models/ weights if big) is NOT committed: .gitignore.
- Code shared and tested (config, data pipeline) lives in src/. The code that is the
  point of a deliverable (models, training, evaluation) is written in the notebook
  itself, visible and commented: the notebook is what is graded and must stand alone.
- Every notebook: Restart & Run All passes, weights reloaded from models/ (no retraining).

## Working agreement (human + Claude Code)
The goal is that the student masters and can explain every step. Therefore:
- One step at a time, in the order of the work items. Do not write several steps ahead.
- After each step: print the checks listed for that step (shapes, counts, one example)
  and wait for the student's validation before moving on.
- Explain every non-trivial line in a short comment; no clever one-liners.
- Never silently change a decision recorded in this file: propose, then update this file.
- Prefer simple and explainable over marginally better and opaque.

## Deliverable 1: binary classification (photo vs other)
Notebook with: TensorFlow code and architecture diagram (parameters, loss,
optimizer); curves of train/test loss and accuracy; analysis of the bias/variance
trade-off; description of improvements (regularization, dropout, early stopping).
Classes: photos vs other images (at least scanned text; then portraits, diagrams,
paintings, which are harder).
Status: done, in the repo.

## Deliverable 2: image denoising with a convolutional auto-encoder
Student subject: a notebook that contains (1) the file loading code, (2) the
denoising code on a subset of noisy images, with explanations, (3) a case study.
Method: convolutional auto-encoders, to improve image quality. The dataset has RGB
images and no noisy images: we generate the noise ourselves from clean photos.

Our interpretation of "case study" (not stated in the subject, say so in the
notebook): show what the denoising gains AND what it costs (loss of fine detail),
with numbers.

### Reuse of D1
Reuse the D1 loading/preprocessing logic (dataset folders, image formats, resizing)
instead of rewriting it, but not the "load everything in RAM" behavior of
src/utils.load_images. src/data.py is the shared data module: avoid two parallel
versions of the same function.

### Decisions
- Data: src/data.py. Gaussian noise, clipped to [0, 1], NOISE_STD 0.1 to 0.2 to start
  (simplifying assumption: real scan noise differs). Train noise regenerated every
  epoch; val/test noise fixed by seed.
- Training: input = noisy image, target = clean image, MSE loss, sigmoid output.
- Baseline model: 3 x (Conv 3x3 same + ReLU + MaxPool 2) with 32-64-128 filters, then
  3 x (Conv + UpSampling 2), final Conv with 3 filters + sigmoid. Check the real
  parameter count with model.summary().
- Improved model: same network with U-Net skip connections (Concatenate between
  encoder and decoder blocks of the same resolution).
- Metrics: PSNR and SSIM (max_val=1.0). Compare on the test set: noisy input,
  a classical filter (Gaussian or median), baseline, U-Net.
- Resizing to 128x128 squashes non-square images: assumed choice, state it.

### Work items (in order)
1. build_autoencoder() in deliverable2.ipynb (done)
2. build_unet() with skip connections, same notebook (done)
3. Training cells in the notebook: early stopping, timestamped checkpoints, saved history, loss curves
4. PSNR/SSIM comparison table
5. Train both models; weights reloadable from models/ without retraining
6. Case study: several noise levels, zoom on a fine detail, difference maps
7. Notebook deliverable2.ipynb (repo root, next to deliverable1.ipynb), Restart & Run All passes with weights reloaded

### Notebook outline
Context / loading / noise and its assumptions / architecture (diagram, role of each
layer, parameter count) / training (loss, optimizer, epochs, curves) / results table
and before-after images / case study / analysis (baseline vs U-Net, overfitting,
bottleneck) / limits and improvements.

### Pitfalls
- Always compare with the noisy input: a model worse than doing nothing is useless.
- U-Net can copy the noise: measure it, do not assume it is better.
- Tune on the validation set, never on the test set.
- Add ds.ignore_errors() in data.py if corrupted images appear.

### If time runs short
Cut extra noise levels first, then the U-Net (keep it as a written recommendation).
Never cut the case study or the numeric comparison of the baseline.

## Deliverable 3: image captioning
Student subject: a notebook containing (1) the full schematic architecture of the
captioning network, stating the CNN type used for preprocessing, (2) a short
description of image and text preprocessing, (3) the code of the CNN and RNN used,
(4) training curves and a display of test examples. It also shows the whole workflow
(D1 + D2 + captioning) as a reproducible pipeline. Dataset: MS COCO.

### Architecture (follows WS_Captioning)
- Encoder: pre-trained CNN (InceptionV3 by default, ImageNet, include_top=False,
  frozen) -> feature grid, reshaped to (regions, channels), cached on disk.
  For InceptionV3: (8, 8, 2048) -> (64, 2048). Trainable part: Dense(256) + ReLU.
- Attention: Bahdanau, units=512 (W1, W2, V). Output: context vector + weights.
- Decoder: Embedding(vocab, 256), GRU(512), Dense(512), Dense(vocab), one word per step.
- Loss: SparseCategoricalCrossentropy(from_logits=True), padding masked. Optimizer: Adam.
- Training: teacher forcing. Inference: greedy decoding (argmax), deterministic.
  Beam search is optional.
- Report layers, units and parameter count from model.summary(), never from hand counts.
- The backbone is a config entry (name, input size, preprocess function, grid shape,
  channels), not hard-coded: number of regions and channels are parameters.

### Data
- MS COCO 2014 train, read from the provided zips (zipfile, no full extraction):
  captions_train2014.json from annotations_trainval2014.zip, images from train2014.zip.
  Check the internal paths of the zip first.
- Seeded random subset (SEED=42), same subset for the whole group. Baseline: 2000
  images, scale up (6000-10000) if time allows. Save the chosen ids in
  data/coco/subset_image_ids.json.
- Split by IMAGE (not by caption), before any other step: 80/10/10 train/val/test.
  Test is never used for tuning. Tokenizer fitted on train only.
- Text: lowercase, <start>/<end>, vocabulary = top K words (justify K with the
  coverage of word frequencies), <unk>, <pad>=0. Cap the caption length at a percentile
  (e.g. 99th) instead of the longest caption.
- Feature cache: cache/<backbone>/<image_id>.npy, one folder per backbone (never mix
  backbones). Tokenizer, max_length and backbone name are saved with the weights.

### Image size decision (OPEN, settled by measurement)
D1/D2 were trained at 128x128. D2 needs a size divisible by 8 (3 pooling stages), 299
is not. Options:
A. Keep D1/D2 at 128, upsample the denoised image to the backbone size.
B. Run everything at the backbone size (retrain D1/D2, ~5.4x more pixels at 299 vs 128).
C. Mixed: D1/D2 on a 128 copy; the captioner gets the original image resized to the
   backbone size when no denoising is needed, the denoised one otherwise.
Measure on the same images: captions direct vs through 128 vs through D2. Choose with
numbers and write the result in the notebook.
Preprocessing: ONE function applies the backbone's own preprocess_input. D1/D2 outputs
are floats in [0, 1]: convert to the range the backbone expects (InceptionV3 expects
pixels 0-255 before preprocess_input, which maps to [-1, 1]).

### Work items (in order, each with its check)
1. Data: JSON + zip, subset, image/caption correspondence. Check: counts, one example
   with its captions, all files present.
2. Split by image. Check: no overlap, sizes.
3. Text preprocessing. Check: one caption words -> ids -> words, length distribution,
   share of <unk>.
4. Image features with the baseline backbone (InceptionV3), tf.data, small batches.
   Check: shape (64, 2048), time for 20 images, then extrapolation.
5. Model (encoder, attention, decoder) in the notebook + diagram.
   Check: one batch through it, shapes printed at each layer, model.summary().
6. Overfit test: train on ~50 images. Check: loss goes near zero; otherwise there is a
   bug (mask, shift, shapes).
7. Full training: train + validation loss per epoch, best checkpoint on val loss,
   history saved, early stopping. Check: both curves.
8. Greedy inference. Check: captions on train images vs test images, typical errors.
9. Attention maps over the image, word by word.
10. Pipeline functions classify(), denoise(), caption(), each callable alone (the jury
    may inject an image at any step), then chained.
11. Measurements: effect of D2 on captions, image size options, error analysis.
12. Optional: backbone comparison (ResNet50, VGG16...) under the rules below; BLEU as an
    indicator only; beam search.

### Backbone comparison rules (optional, after step 8)
Same subset, split, tokenizer, epochs and seed; only the backbone changes. One cache
folder per backbone, the backbone's own preprocess_input and input size. Judge on val
loss and on captions, not on one run. Limit to one or two comparisons.

### Pitfalls
- Features are computed once: any change of backbone, image size or preprocessing
  means recomputing them.
- The workshop samples words at inference (non-reproducible demo): use argmax.
- The workshop saves checkpoints at epochs 0, 5, 10, 15 only, derives start_epoch from
  the checkpoint file name and does not save the tokenizer: save best/last explicitly,
  store the epoch yourself.
- Do not report only the train loss.
- COCO 2014 and 2017 file names differ; keep the 2014 naming.
- Distribution shift: the captioner is trained on clean full-resolution COCO images,
  but at inference the image went through 128x128 and D2. Measure it, discuss it.

### Defense demo requirements
- Feature extraction of the pipeline and loading of the models are done before the
  presentation; in the demo only the last cell is adapted to the provided dataset.
- A non-photo image chosen by the jury must go classification -> denoising ->
  captioning; if a step fails, an image can be injected directly into the next step.
- Captions are compared with the expected ones in captioning.txt.

### If time runs short
Cut backbone comparison, beam search, BLEU, attention maps (in this order). Never cut:
working end-to-end pipeline, validation loss, qualitative analysis, the measurement of
the image size / D2 effect.

## Oral preparation (likely topics, not an official list)
How neural networks, CNNs, RNNs and auto-encoders work; softmax, ReLU, dropout,
batch normalization; epoch/batch/iteration; normalization; overfitting and
underfitting; learning rate; transfer learning and fine-tuning; backpropagation;
for D2: why input is noisy and target clean, why the split comes before the noise,
why the test noise is fixed.
For D3: why the backbone is frozen and features cached; teacher forcing vs inference
(exposure bias); the padding mask; softmax and cross-entropy; GRU vs simple RNN
(vanishing gradient); what attention weights mean; what the hidden state is; why the
last Dense layer is the heaviest; why this model has no dropout or batch norm (or add
one and justify it).