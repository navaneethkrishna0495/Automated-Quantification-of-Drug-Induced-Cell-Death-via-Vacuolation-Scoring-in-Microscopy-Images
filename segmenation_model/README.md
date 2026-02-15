# Segmentation Model (Mask R-CNN)

This directory contains the trained Mask R-CNN (Detectron2) model used for cell instance segmentation.

## Download the Model Weights

The `model_final.pth` file is too large for Git. Download the compressed archive from Google Drive:

**[Download trained_model.gz](https://drive.google.com/file/d/1vYuGhIE8s-02iqVymYC0KQ3fajFzb7yq/view)**

### Steps

1. Download `trained_model.gz` from the link above
2. Extract the contents into this `segmenation_model/` directory:

```bash
# Linux / WSL
gzip -d trained_model.gz
# or
tar -xzf trained_model.gz -C segmenation_model/
```

3. After extraction, the directory should look like:

```
segmenation_model/
├── README.md
├── config.yaml
├── last_checkpoint
├── metrics.json
└── model_final.pth    <-- extracted from trained_model.gz
```

> **Important:** Make sure `model_final.pth` ends up directly inside `segmenation_model/`. The inference script expects it at this exact path.

## Model Details

- **Architecture:** Mask R-CNN with ResNet-50 + FPN backbone (`mask_rcnn_R_50_FPN_3x`)
- **Classes:** 1 (cell)
- **Framework:** Detectron2
- **Training config:** See `config.yaml` for full training configuration
