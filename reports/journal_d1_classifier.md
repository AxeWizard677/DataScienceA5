# Training journal: d1_classifier

## 01_mlp_baseline: **BASELINE**
- **Hypothesis:** A fully-connected network on flattened 64x64 pixels can separate photos from other images
- **Config:** `{"builder": "mlp", "builder_kwargs": {}, "epochs": 25, "lr": 0.001, "batch_size": 64, "img_size": 160, "train_sources": null, "early_stopping": false, "patience": 6}`
- **Evaluation:** val_accuracy=0.7966, val_precision=0.5949, val_recall=0.8406, val_f1=0.6967, val_roc_auc=0.8906, train_accuracy=0.8510, train_precision=0.6670, train_recall=0.9261, train_f1=0.7754, train_roc_auc=0.9467
- **Train/val loss gap:** 0.1279 (best val epoch 15/25)
- **Lesson / next step:** High bias: an MLP on flattened pixels cannot fit the task (train 85%). Paintings rejected only 52% of the time -> use a CNN

## 02_cnn_easy_sources_only: **FAILURE**
- **Hypothesis:** A simple CNN trained on photo vs text/schematics generalises to paintings and sketches
- **Config:** `{"builder": "simple_cnn", "builder_kwargs": {}, "epochs": 15, "lr": 0.001, "batch_size": 32, "img_size": 160, "train_sources": ["Photo", "Text", "Schematics"], "early_stopping": false, "patience": 6}`
- **Evaluation:** val_accuracy=0.7532, val_precision=0.5303, val_recall=0.9800, val_f1=0.6882, val_roc_auc=0.8724, train_accuracy=0.9966, train_precision=0.9952, train_recall=0.9964, train_f1=0.9958, train_roc_auc=0.9999
- **Train/val loss gap:** 2.8996 (best val epoch 4/15)
- **Lesson / next step:** Trained without paintings/sketches, it calls 84% of paintings photos: a model only learns distinctions present in its training data -> train on all sources

## 03_cnn_all_sources_no_reg: **IMPROVEMENT**
- **Hypothesis:** The same CNN trained on all sources learns to reject paintings and sketches
- **Config:** `{"builder": "simple_cnn", "builder_kwargs": {}, "epochs": 25, "lr": 0.001, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": false, "patience": 6}`
- **Evaluation:** val_accuracy=0.9084, val_precision=0.8531, val_recall=0.8099, val_f1=0.8309, val_roc_auc=0.9605, train_accuracy=0.9990, train_precision=0.9988, train_recall=0.9976, train_f1=0.9982, train_roc_auc=1.0000
- **Train/val loss gap:** 0.6081 (best val epoch 3/25)
- **Lesson / next step:** Adding hard sources: F1 0.69 -> 0.83, but train 99.9% vs val 90.8%: high variance (val loss rises while train loss falls) -> regularise

## 04_cnn_regularized: **FAILURE**
- **Hypothesis:** Augmentation + BatchNorm + L2 + GAP + dropout + early stopping close E3's generalisation gap
- **Config:** `{"builder": "regularized_cnn", "builder_kwargs": {"l2": 0.0001, "dropout": 0.4}, "epochs": 40, "lr": 0.001, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 6, "augmentation": "flip/rot/zoom/translate"}`
- **Evaluation:** val_accuracy=0.8927, val_precision=0.8038, val_recall=0.8119, val_f1=0.8078, val_roc_auc=0.9519, train_accuracy=0.8921, train_precision=0.7918, train_recall=0.8296, train_f1=0.8102, train_roc_auc=0.9514
- **Train/val loss gap:** 0.1763 (best val epoch 13/19)
- **Lesson / next step:** Gap closed (train 89.2 / val 89.3) but F1 0.808 < E3: over-regularised, bias went up (trade-off). Defaults are not tuned values -> E4b sweep; from-scratch CNN capped ~90% -> transfer learning

## 05_mobilenetv2_frozen: **IMPROVEMENT**
- **Hypothesis:** ImageNet features (frozen MobileNetV2) separate photos from paintings better than a CNN from scratch
- **Config:** `{"builder": "mobilenet_frozen", "builder_kwargs": {}, "epochs": 15, "lr": 0.001, "batch_size": 64, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 4, "backbone": "MobileNetV2/imagenet", "trainable_layers": "head only"}`
- **Evaluation:** val_accuracy=0.9690, val_precision=0.9210, val_recall=0.9720, val_f1=0.9458, val_roc_auc=0.9951, train_accuracy=0.9762, train_precision=0.9393, train_recall=0.9774, train_f1=0.9579, train_roc_auc=0.9973
- **Train/val loss gap:** -0.0515 (best val epoch 9/13)
- **Lesson / next step:** Transfer learning: F1 0.83 -> 0.946, AUC 0.995 with only 1,281 trained weights; low bias and low variance. Remaining errors: realistic paintings -> fine-tune the top layers

## 06_mobilenetv2_finetuned: **BEST**
- **Hypothesis:** Fine-tuning the top 40 MobileNetV2 layers at a low learning rate improves on E5
- **Config:** `{"builder": "mobilenet_finetune", "builder_kwargs": {"from_model": "/home/utilisateur/Bureau/DataScienceA5/runs/d1_classifier/05_mobilenetv2_frozen/model.keras", "n_layers": 40}, "epochs": 15, "lr": 1e-05, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 4, "backbone": "MobileNetV2/imagenet", "trainable_layers": "last 40 (BatchNorm frozen)", "init": "E5 weights"}`
- **Evaluation:** val_accuracy=0.9787, val_precision=0.9408, val_recall=0.9853, val_f1=0.9625, val_roc_auc=0.9984, train_accuracy=0.9831, train_precision=0.9529, train_recall=0.9881, train_f1=0.9702, train_roc_auc=0.9991
- **Train/val loss gap:** 0.0069 (best val epoch 3/7)
- **Lesson / next step:** Test F1 0.962 / AUC 0.997. Weakness: grayscale photos (recall 96.2 -> 91.9%), old sepia prints missed. Some errors are label noise in the archives

## 07_reg_d02_l2_1e-5: **FAILURE**
- **Hypothesis:** Lighter regularisation (dropout 0.2, L2 1e-05) than E4 lowers bias while keeping the gap small
- **Config:** `{"builder": "regularized_cnn", "epochs": 30, "lr": 0.001, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 5, "augmentation": "flip/rot/zoom/translate", "builder_kwargs": {"dropout": 0.2, "l2": 1e-05}}`
- **Evaluation:** val_accuracy=0.8623, val_precision=0.7079, val_recall=0.8586, val_f1=0.7760, val_roc_auc=0.9368, train_accuracy=0.8679, train_precision=0.7099, train_recall=0.8868, train_f1=0.7886, train_roc_auc=0.9393
- **Train/val loss gap:** 0.2057 (best val epoch 4/9)
- **Lesson / next step:** Lighter reg. did NOT lower bias: early stopping fired at epoch 9 on oscillating val loss (best ep 4), F1 0.776

## 08_reg_d00_l2_1e-5: **FAILURE**
- **Hypothesis:** Lighter regularisation (dropout 0.0, L2 1e-05) than E4 lowers bias while keeping the gap small
- **Config:** `{"builder": "regularized_cnn", "epochs": 30, "lr": 0.001, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 5, "augmentation": "flip/rot/zoom/translate", "builder_kwargs": {"dropout": 0.0, "l2": 1e-05}}`
- **Evaluation:** val_accuracy=0.8396, val_precision=0.6779, val_recall=0.8059, val_f1=0.7364, val_roc_auc=0.9153, train_accuracy=0.8401, train_precision=0.6738, train_recall=0.8224, train_f1=0.7407, train_roc_auc=0.9177
- **Train/val loss gap:** 0.0664 (best val epoch 3/8)
- **Lesson / next step:** No dropout: even earlier stop (epoch 8), F1 0.736 -> dropout also stabilises training

## 09_reg_d04_l2_0: **IMPROVEMENT**
- **Hypothesis:** Lighter regularisation (dropout 0.4, L2 0.0) than E4 lowers bias while keeping the gap small
- **Config:** `{"builder": "regularized_cnn", "epochs": 30, "lr": 0.001, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 5, "augmentation": "flip/rot/zoom/translate", "builder_kwargs": {"dropout": 0.4, "l2": 0.0}}`
- **Evaluation:** val_accuracy=0.9149, val_precision=0.8316, val_recall=0.8699, val_f1=0.8503, val_roc_auc=0.9694, train_accuracy=0.9186, train_precision=0.8291, train_recall=0.8903, train_f1=0.8586, train_roc_auc=0.9719
- **Train/val loss gap:** 0.0753 (best val epoch 10/15)
- **Lesson / next step:** Dropout 0.4 without L2: best from-scratch CNN, F1 0.850 / AUC 0.969, gap 0.4 pt -> L2 was over-constraining. Val loss still oscillates -> LR schedule (E4c)

## 10_reg_d04_l2_0_lr_schedule: **IMPROVEMENT**
- **Hypothesis:** With E4b-3 regularisation, halving the LR on plateau and patience 10 stabilise training and lower bias further
- **Config:** `{"builder": "regularized_cnn", "builder_kwargs": {"dropout": 0.4, "l2": 0.0}, "epochs": 40, "lr": 0.001, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 10, "reduce_lr": true, "augmentation": "flip/rot/zoom/translate"}`
- **Evaluation:** val_accuracy=0.9160, val_precision=0.7982, val_recall=0.9340, val_f1=0.8607, val_roc_auc=0.9776, train_accuracy=0.9345, train_precision=0.8294, train_recall=0.9619, train_f1=0.8907, train_roc_auc=0.9847
- **Train/val loss gap:** 0.0704 (best val epoch 22/32)
- **Lesson / next step:** LR halved on plateau (5x) + patience 10: val loss stabilised (0.21-0.23 vs 0.37-0.76), best epoch 22, F1 0.861 / AUC 0.978 = best from-scratch CNN; still far below transfer learning (0.963)
