# Training journal: d1_classifier_v2

## 01_v1_frozen: **BASELINE**
- **Hypothesis:** E5 recipe (frozen MobileNetV2) on the complete dataset
- **Config:** `{"builder": "mobilenet_frozen", "builder_kwargs": {}, "epochs": 15, "lr": 0.001, "batch_size": 64, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 4, "split_file": "d1_split_v2.csv"}`
- **Evaluation:** val_accuracy=0.9636, val_precision=0.8864, val_recall=0.9787, val_f1=0.9302, val_roc_auc=0.9948, train_accuracy=0.9684, train_precision=0.8978, train_recall=0.9845, train_f1=0.9392, train_roc_auc=0.9959
- **Train/val loss gap:** -0.0429 (best val epoch 8/12)
- **Lesson / next step:** Intermediate stage (frozen MobileNetV2), starting point of V1's fine-tuning; same AUC as E5 frozen, threshold shifted by the new class balance

## 02_v1_finetune40: **BASELINE**
- **Hypothesis:** E6 recipe (fine-tune 40 layers, lr 1e-5) on the complete dataset reaches at least E6's level
- **Config:** `{"builder": "mobilenet_finetune", "builder_kwargs": {"from_model": "/home/utilisateur/Bureau/DataScienceA5/runs/d1_classifier_v2/01_v1_frozen/model.keras", "n_layers": 40}, "epochs": 15, "lr": 1e-05, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 4, "split_file": "d1_split_v2.csv"}`
- **Evaluation:** val_accuracy=0.9854, val_precision=0.9626, val_recall=0.9793, val_f1=0.9709, val_roc_auc=0.9984, train_accuracy=0.9911, train_precision=0.9753, train_recall=0.9893, train_f1=0.9822, train_roc_auc=0.9996
- **Train/val loss gap:** 0.0194 (best val epoch 4/8)
- **Lesson / next step:** E6 recipe on complete data: F1 0.971, 82 errors on fixed set (E6: 141) -> new reference

## 03_v2_gray_frozen: **BASELINE**
- **Hypothesis:** Frozen MobileNetV2 with 25% random-grayscale augmentation
- **Config:** `{"builder": "mobilenet_frozen", "builder_kwargs": {"grayscale": 0.25}, "epochs": 15, "lr": 0.001, "batch_size": 64, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 4, "split_file": "d1_split_v2.csv", "augmentation": "flip/rot/zoom/translate + grayscale 0.25"}`
- **Evaluation:** val_accuracy=0.9664, val_precision=0.8990, val_recall=0.9740, val_f1=0.9350, val_roc_auc=0.9947, train_accuracy=0.9705, train_precision=0.9083, train_recall=0.9797, train_f1=0.9427, train_roc_auc=0.9955
- **Train/val loss gap:** -0.0790 (best val epoch 15/15)
- **Lesson / next step:** Intermediate stage, starting point of V2 and V3; better than V1 frozen (192 vs 206 errors on the fixed set)

## 04_v2_gray_finetune40: **FAILURE**
- **Hypothesis:** Grayscale augmentation removes the colour shortcut without hurting accuracy
- **Config:** `{"builder": "mobilenet_finetune", "builder_kwargs": {"from_model": "/home/utilisateur/Bureau/DataScienceA5/runs/d1_classifier_v2/03_v2_gray_frozen/model.keras", "n_layers": 40}, "epochs": 15, "lr": 1e-05, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 4, "split_file": "d1_split_v2.csv", "augmentation": "flip/rot/zoom/translate + grayscale 0.25"}`
- **Evaluation:** val_accuracy=0.9830, val_precision=0.9480, val_recall=0.9853, val_f1=0.9663, val_roc_auc=0.9986, train_accuracy=0.9917, train_precision=0.9710, train_recall=0.9964, train_f1=0.9835, train_roc_auc=0.9998
- **Train/val loss gap:** 0.0216 (best val epoch 7/11)
- **Lesson / next step:** Grayscale aug: grayscale photo recall 95.9 -> 97.0% (goal reached) but F1 0.966 < V1 0.971

## 05_v3_gray_ls_finetune60: **IMPROVEMENT**
- **Hypothesis:** Label smoothing + 60 unfrozen layers + LR schedule improve on V2
- **Config:** `{"builder": "mobilenet_finetune", "builder_kwargs": {"from_model": "/home/utilisateur/Bureau/DataScienceA5/runs/d1_classifier_v2/03_v2_gray_frozen/model.keras", "n_layers": 60}, "epochs": 20, "lr": 1e-05, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 5, "split_file": "d1_split_v2.csv", "reduce_lr": true, "label_smoothing": 0.05, "augmentation": "flip/rot/zoom/translate + grayscale 0.25"}`
- **Evaluation:** val_accuracy=0.9868, val_precision=0.9634, val_recall=0.9840, val_f1=0.9736, val_roc_auc=0.9979, train_accuracy=0.9947, train_precision=0.9812, train_recall=0.9976, train_f1=0.9894, train_roc_auc=0.9999
- **Train/val loss gap:** 0.0080 (best val epoch 12/17)
- **Lesson / next step:** LS + 60 layers + LR schedule from V2 frozen: F1 0.974, 76 errors on fixed set; best at 160 px

## 06_v4_gray_frozen_224: **BASELINE**
- **Hypothesis:** Frozen MobileNetV2 at 224 px with grayscale augmentation
- **Config:** `{"builder": "mobilenet_frozen", "builder_kwargs": {"grayscale": 0.25}, "epochs": 15, "lr": 0.001, "batch_size": 64, "img_size": 224, "train_sources": null, "early_stopping": true, "patience": 4, "split_file": "d1_split_v2.csv", "augmentation": "flip/rot/zoom/translate + grayscale 0.25"}`
- **Evaluation:** val_accuracy=0.9732, val_precision=0.9196, val_recall=0.9773, val_f1=0.9476, val_roc_auc=0.9968, train_accuracy=0.9770, train_precision=0.9309, train_recall=0.9797, train_f1=0.9547, train_roc_auc=0.9972
- **Train/val loss gap:** -0.0556 (best val epoch 9/13)
- **Lesson / next step:** Intermediate stage at 224 px, starting point of V4; best frozen stage (155 errors on the fixed set)

## 07_v4_gray_ls_finetune60_224: **BEST**
- **Hypothesis:** V3's recipe at 224 px improves on realistic paintings
- **Config:** `{"builder": "mobilenet_finetune", "builder_kwargs": {"from_model": "/home/utilisateur/Bureau/DataScienceA5/runs/d1_classifier_v2/06_v4_gray_frozen_224/model.keras", "n_layers": 60}, "epochs": 20, "lr": 1e-05, "batch_size": 32, "img_size": 224, "train_sources": null, "early_stopping": true, "patience": 5, "split_file": "d1_split_v2.csv", "reduce_lr": true, "label_smoothing": 0.05, "augmentation": "flip/rot/zoom/translate + grayscale 0.25"}`
- **Evaluation:** val_accuracy=0.9902, val_precision=0.9724, val_recall=0.9887, val_f1=0.9805, val_roc_auc=0.9986, train_accuracy=0.9979, train_precision=0.9941, train_recall=0.9976, train_f1=0.9958, train_roc_auc=1.0000
- **Train/val loss gap:** 0.0085 (best val epoch 18/20)
- **Lesson / next step:** V3 at 224 px: val acc 99.0%, F1 0.980, grayscale recall 98.5%; test acc 99.09%

## 08_own_cnn_e4c_recipe: **FAILURE**
- **Hypothesis:** Our best part-1 from-scratch recipe (E4c) on the complete dataset: fair comparison with transfer learning
- **Config:** `{"builder": "regularized_cnn", "builder_kwargs": {"dropout": 0.4, "l2": 0.0}, "epochs": 40, "lr": 0.001, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 10, "split_file": "d1_split_v2.csv", "reduce_lr": true, "augmentation": "flip/rot/zoom/translate"}`
- **Evaluation:** val_accuracy=0.9262, val_precision=0.8052, val_recall=0.9266, val_f1=0.8617, val_roc_auc=0.9788, train_accuracy=0.9291, train_precision=0.8034, train_recall=0.9452, train_f1=0.8686, train_roc_auc=0.9840
- **Train/val loss gap:** 0.0759 (best val epoch 24/34)
- **Lesson / next step:** Best part-1 own recipe on complete data: 435 errors / 4,523 (90.4%), 49 min: from-scratch training can't match transfer learning

## 09_own_resnet: **IMPROVEMENT**
- **Hypothesis:** A modern from-scratch architecture (small ResNet, 2.8M params, comparable to MobileNetV2) closes part of the gap
- **Config:** `{"builder": "residual_cnn", "builder_kwargs": {"dropout": 0.4}, "epochs": 40, "lr": 0.001, "batch_size": 32, "img_size": 160, "train_sources": null, "early_stopping": true, "patience": 10, "split_file": "d1_split_v2.csv", "reduce_lr": true, "augmentation": "flip/rot/zoom/translate"}`
- **Evaluation:** val_accuracy=0.9363, val_precision=0.8288, val_recall=0.9366, val_f1=0.8794, val_roc_auc=0.9835, train_accuracy=0.9468, train_precision=0.8443, train_recall=0.9631, train_f1=0.8998, train_roc_auc=0.9904
- **Train/val loss gap:** 0.0763 (best val epoch 37/40)
- **Lesson / next step:** Own ResNet: 373 errors (-14% vs O1), fastest inference (821 img/s GPU, 118 CPU), half the training memory; still 4.5x more errors than V1
