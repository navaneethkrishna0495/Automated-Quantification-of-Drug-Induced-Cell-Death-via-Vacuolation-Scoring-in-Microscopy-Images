"""
Reproducible inference: image -> Mask R-CNN segmentation (colored by cell) -> XGBoost -> drug efficacy score.
"""
import os
import sys
import argparse

# Fail early with a clear message if torch cannot be loaded (common on Windows: DLL / VC++ Redistributable)
try:
    import cv2
    import numpy as np
    import torch
    import joblib
except ImportError as e:
    err = str(e)
    if "torch" in err or "DLL" in err or "_C" in err or "could not be found" in err:
        print("PyTorch failed to load. On Windows this is often fixed by:")
        print("  1. Install Microsoft Visual C++ Redistributable (latest): https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("  2. Or use Python 3.10 or 3.11 in a venv (PyTorch/Detectron2 are well-tested on those).")
    raise

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis


# Score mapping: class index -> numeric cell death score (paper: 0, 0.3, 0.6, 0.9, 1.0)
SCORE_MAP = np.array([0.0, 0.3, 0.6, 0.9, 1.0])

# Project root (directory containing predict.py); weights paths are relative to this
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_features(region_image):
    """Full handcrafted feature extractor (ref_code parity): GLCM, hist, LBP, shape, edge, Fourier, Gabor. Returns vector in sorted key order."""
    gray_image = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY)
    if gray_image.dtype != np.uint8:
        gray_image = np.clip(gray_image, 0, 255).astype(np.uint8)
    features = {}
    # 1. GLCM
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    for prop in ['energy', 'contrast', 'correlation', 'homogeneity', 'ASM', 'dissimilarity']:
        features[prop] = float(np.round(graycoprops(glcm, prop)[0, 0], 2))
    features['glcm_entropy'] = float(np.round(-np.sum(np.nan_to_num(glcm * np.log(glcm + 1e-10))), 2))
    # 2. Histogram
    pix = gray_image.flatten()
    features['hist_mean'] = float(pix.mean())
    features['hist_std'] = float(pix.std())
    features['hist_skewness'] = float(skew(pix))
    features['hist_kurtosis'] = float(kurtosis(pix))
    # 3. LBP
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 12), range=(0, 11))
    hist = hist.astype(float) / (hist.sum() + 1e-6)
    for i, v in enumerate(hist):
        features[f'lbp_{i}'] = float(v)
    # 4. Shape
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        x, y, w, h = cv2.boundingRect(c)
        features['area'] = float(area)
        features['perimeter'] = float(perimeter)
        features['solidity'] = float(area / hull_area if hull_area > 0 else 0)
        features['aspect_ratio'] = float(w / h if h > 0 else 0)
    else:
        for k in ['area', 'perimeter', 'solidity', 'aspect_ratio']:
            features[k] = 0.0
    # 5. Edge density
    edges = cv2.Canny(gray_image, 100, 200)
    features['edge_density'] = float((edges > 0).sum() / (gray_image.size + 1e-10))
    # 6. Fourier
    f = np.fft.fftshift(np.fft.fft2(gray_image))
    mag = np.abs(f)
    features['fourier_mean'] = float(mag.mean())
    features['fourier_std'] = float(mag.std())
    # 7. Gabor
    angles = [0, np.pi / 4, np.pi / 2]
    for i, angle in enumerate(angles):
        k = cv2.getGaborKernel((21, 21), 5, angle, 10, 0.5, 0, cv2.CV_32F)
        filt = cv2.filter2D(gray_image, cv2.CV_32F, k)
        features[f'gabor_mean_{i}'] = float(filt.mean())
        features[f'gabor_std_{i}'] = float(filt.std())
    return np.array([features[k] for k in sorted(features.keys())], dtype=np.float64)


def extract_deep_feature_for_box(model, image, box, device):
    """Extract 1024-d deep feature for one box from Mask R-CNN backbone + box head."""
    h, w = image.shape[:2]
    inst = Instances((h, w))
    inst.pred_boxes = Boxes(torch.tensor(box[None, :], dtype=torch.float32, device=device))
    inst = inst.to(device)
    inp = {
        "image": torch.as_tensor(image.transpose(2, 0, 1).astype(np.float32)).to(device),
        "height": h,
        "width": w,
    }
    with torch.no_grad():
        imgs = model.preprocess_image([inp])
        feats = model.backbone(imgs.tensor)
        box_feats = model.roi_heads.box_pooler(
            [feats[f] for f in model.roi_heads.in_features],
            [inst.pred_boxes]
        )
        box_flat = model.roi_heads.box_head(box_feats)
    return box_flat[0].cpu().numpy()


def load_segmenter(weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
    return cfg, predictor, predictor.model


def main(img_path, out_seg=None):
    mask_weights = os.path.join(SCRIPT_DIR, "weights", "model_final.pth")
    xgb_path = os.path.join(SCRIPT_DIR, "weights", "xgb_classifier_deep.pkl")

    # Create results folder on run (relative to project root)
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    img_stem = os.path.splitext(os.path.basename(img_path))[0]

    if not os.path.isfile(mask_weights):
        print(f"Error: Mask R-CNN weights not found at {mask_weights}")
        return
    if not os.path.isfile(xgb_path):
        print(f"Error: XGBoost model not found at {xgb_path}")
        return

    cfg, predictor, model = load_segmenter(mask_weights)
    clf = joblib.load(xgb_path)
    device = cfg.MODEL.DEVICE

    if not os.path.isfile(img_path):
        print(f"Error: image file not found: {img_path}")
        return
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: could not read image {img_path}")
        return

    outputs = predictor(img)
    instances = outputs["instances"].to("cpu")
    n_cells = len(instances)

    # Segmentation visualization: each cell different color
    metadata = {"thing_classes": ["cell"]}
    v = Visualizer(img[:, :, ::-1], scale=0.8, metadata=metadata)
    out_vis = v.draw_instance_predictions(instances)
    seg_image = out_vis.get_image()[:, :, ::-1]

    if out_seg is None:
        out_seg = os.path.join(results_dir, f"{img_stem}_segmentation.png")
    out_dir = os.path.dirname(out_seg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_seg, seg_image)
    print(f"Segmentation saved: {out_seg}")
    print(f"Number of cells detected: {n_cells}")

    if n_cells == 0:
        score_path = os.path.join(results_dir, f"{img_stem}_score.txt")
        with open(score_path, "w") as f:
            f.write(f"Input image: {img_path}\n")
            f.write("Number of cells detected: 0\n")
            f.write("Drug efficacy score: N/A (no cells)\n")
        print(f"Score saved: {score_path}")
        print("No cells detected. Drug efficacy score not computed.")
        return

    boxes = instances.pred_boxes.tensor.numpy()
    per_cell_scores = []

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        hfeat = extract_features(crop)
        dfeat = extract_deep_feature_for_box(model, img, box, device)
        feats = np.hstack([hfeat, dfeat]).reshape(1, -1)

        pred_idx = int(clf.predict(feats)[0])
        if hasattr(clf, 'classes_') and hasattr(clf.classes_, '__len__'):
            try:
                score = float(clf.classes_[pred_idx])
            except (ValueError, IndexError, TypeError):
                score = SCORE_MAP[pred_idx] if 0 <= pred_idx < len(SCORE_MAP) else float(pred_idx)
        else:
            score = SCORE_MAP[pred_idx] if 0 <= pred_idx < len(SCORE_MAP) else float(pred_idx)
        per_cell_scores.append(score)

    if len(per_cell_scores) == 0:
        print("No valid cell crops. Drug efficacy score not computed.")
        return

    per_cell_scores = np.array(per_cell_scores)
    drug_efficacy_score = float(np.mean(per_cell_scores))

    # Save score to results folder
    score_path = os.path.join(results_dir, f"{img_stem}_score.txt")
    with open(score_path, "w") as f:
        f.write(f"Input image: {img_path}\n")
        f.write(f"Number of cells detected: {len(per_cell_scores)}\n")
        f.write(f"Per-cell death scores: {per_cell_scores.tolist()}\n")
        f.write(f"Drug efficacy score (mean cell death score): {drug_efficacy_score:.4f}\n")
    print(f"Score saved: {score_path}")

    print(f"Per-cell death scores: {per_cell_scores.tolist()}")
    print(f"Final drug efficacy score (mean cell death score): {drug_efficacy_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mask R-CNN + XGBoost inference and output segmentation image and drug efficacy score.")
    parser.add_argument("--image", required=True, help="Path to input microscopy image")
    parser.add_argument("--out_seg", default=None, help="Path for segmentation output image (default: <image_stem>_segmentation.png)")
    args = parser.parse_args()
    main(args.image, args.out_seg)