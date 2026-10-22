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


## Shared technical conventions
- Images: RGB, 128x128 for D1/D2 (D3 needs 299x299 for InceptionV3: decide where the
  resize happens), float32 in [0, 1] when fed to a model.
- Memory: some teammates have a 2 GB GPU. Small batches, tf.data, uint8 in memory.
- Reproducibility: fixed seeds, pinned versions, relative paths only.
- Code shared and tested (config, data pipeline) lives in src/. The code that is the
  point of a deliverable (models, training, evaluation) is written in the notebook
  itself, visible and commented: the notebook is what is graded and must stand alone.

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
There is no official evaluation grid.

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
Notebook with: full architecture diagram (CNN encoder + RNN decoder, layers, cells,
role of each); short description of image and text preprocessing; code of the CNN and
RNN; training curves; a few captioned test images. Dataset: MS COCO.
Images go through a pre-trained InceptionV3 first, features are stored on disk.
Text preprocessing: tokenization, most frequent words, word indexes, padding.
The notebook also shows the whole workflow including D1 and D2.

## Oral preparation (likely topics, not an official list)
How neural networks, CNNs, RNNs and auto-encoders work; softmax, ReLU, dropout,
batch normalization; epoch/batch/iteration; normalization; overfitting and
underfitting; learning rate; transfer learning and fine-tuning; backpropagation;
for D2: why input is noisy and target clean, why the split comes before the noise,
why the test noise is fixed.