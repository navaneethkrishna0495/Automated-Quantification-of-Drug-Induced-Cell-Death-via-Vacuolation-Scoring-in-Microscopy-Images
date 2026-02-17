import os
import sys
import argparse

# Fail early with a clear message if torch cannot be loaded
try:
    import cv2
    import numpy as np
    import torch
    import joblib
except ImportError as e:
    err = str(e)
    if "torch" in err or "DLL" in err or "_C" in err or "could not be found" in err:
        print("PyTorch failed to load. On Windows this is often fixed by:")
        print(
            "  1. Install Microsoft Visual C++ Redistributable (latest): "
            "https://aka.ms/vs/17/release/vc_redist.x64.exe"
        )
        print(
            "  2. Or use Python 3.10 or 3.11 in a venv "
            "(PyTorch/Detectron2 are well-tested on those)."
        )
    raise

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis


REVERSE_MAP = {0: 0.0, 1: 0.3, 2: 0.6, 3: 0.9, 4: 1.0}


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
# Mask R-CNN weights live inside segmenation_model/
MASK_RCNN_WEIGHTS = os.path.join(SCRIPT_DIR, "segmenation_model", "model_final.pth")

# XGBoost model
XGBOOST_MODEL = os.path.join(SCRIPT_DIR, "xgboost", "xgboost_cell_model.pkl")


# ---------------------------------------------------------------------------
# Feature extraction  (handcrafted) 
# ---------------------------------------------------------------------------
def extract_handcrafted(crop):
    features = {}

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    # 1. GLCM (multi-angle)
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    for prop in ["contrast", "correlation", "energy", "homogeneity", "dissimilarity"]:
        features[f"glcm_{prop}"] = np.mean(graycoprops(glcm, prop))

    features["glcm_entropy"] = -np.sum(glcm * np.log(glcm + 1e-10))

    # 2. Statistical
    pix = gray.flatten()

    features["mean"] = np.mean(pix)
    features["std"] = np.std(pix)
    features["skew"] = skew(pix)
    features["kurtosis"] = kurtosis(pix)

    # 3. LBP (uniform)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")

    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))

    hist = hist / (hist.sum() + 1e-6)

    for i, val in enumerate(hist):
        features[f"lbp_{i}"] = val

    # 4. Shape features
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        c = max(cnts, key=cv2.contourArea)

        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)

        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)

        x, y, w, h = cv2.boundingRect(c)

        features["area"] = area
        features["perimeter"] = perim
        features["aspect_ratio"] = w / (h + 1e-6)
        features["solidity"] = area / (hull_area + 1e-6)
    else:
        features["area"] = 0
        features["perimeter"] = 0
        features["aspect_ratio"] = 0
        features["solidity"] = 0

    # 5. Edge density
    edges = cv2.Canny(gray, 100, 200)
    features["edge_density"] = np.sum(edges > 0) / gray.size

    # 6. Fourier
    f = np.fft.fft2(gray)
    mag = np.abs(f)

    features["fourier_mean"] = np.mean(mag)
    features["fourier_std"] = np.std(mag)

    # 7. Gabor (multi-angle)
    gabor_vals = []

    for theta in [0, np.pi / 4, np.pi / 2]:
        kernel = cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5)
        gabor = cv2.filter2D(gray, cv2.CV_32F, kernel)
        gabor_vals.append(np.mean(gabor))

    features["gabor_mean"] = np.mean(gabor_vals)
    features["gabor_std"] = np.std(gabor_vals)

    return np.array(list(features.values()), dtype=float)


# ---------------------------------------------------------------------------
# Feature extraction  (deep — from Mask R-CNN backbone)
# ---------------------------------------------------------------------------
def extract_deep(model, device, image, box):
    h, w = image.shape[:2]

    inst = Instances((h, w))
    inst.pred_boxes = Boxes(torch.tensor([box]).float()).to(device)

    inp = {
        "image": torch.as_tensor(image.transpose(2, 0, 1)).float().to(device),
        "height": h,
        "width": w,
    }

    with torch.no_grad():
        images = model.preprocess_image([inp])
        features = model.backbone(images.tensor)

        box_features = model.roi_heads.box_pooler(
            [features[f] for f in model.roi_heads.in_features],
            [inst.pred_boxes],
        )

        deep_feats = model.roi_heads.box_head(box_features)

    return deep_feats.squeeze().cpu().numpy()


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_mask_rcnn(weights_path):
    """Load Mask R-CNN (Detectron2) with the trained cell-segmentation weights."""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = DefaultPredictor(cfg)
    print(f"Mask R-CNN loaded  (device={cfg.MODEL.DEVICE})")
    return cfg, predictor, predictor.model


def load_xgboost(model_path):
    """Load the XGBoost vacuolation classifier."""
    clf = joblib.load(model_path)
    print("XGBoost loaded")
    return clf


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------
def main(img_path, out_seg=None):
    # ---- resolve & validate model paths ----
    mask_weights = os.path.normpath(MASK_RCNN_WEIGHTS)
    xgb_path = os.path.normpath(XGBOOST_MODEL)

    if not os.path.isfile(mask_weights):
        print(f"Error: Mask R-CNN weights not found at {mask_weights}")
        return
    if not os.path.isfile(xgb_path):
        print(f"Error: XGBoost model not found at {xgb_path}")
        return

    # ---- results directory ----
    results_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    img_stem = os.path.splitext(os.path.basename(img_path))[0]

    # ---- load models ----
    cfg, predictor, model = load_mask_rcnn(mask_weights)
    xgb = load_xgboost(xgb_path)
    device = cfg.MODEL.DEVICE

    # ---- read image ----
    if not os.path.isfile(img_path):
        print(f"Error: image file not found: {img_path}")
        return
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error: could not read image {img_path}")
        return

    # ---- run Mask R-CNN ----
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    n_cells = len(instances)

    # ---- save segmentation visualisation ----
    metadata = {"thing_classes": ["cell"]}
    vis = Visualizer(image[:, :, ::-1], scale=0.8, metadata=metadata)
    out_vis = vis.draw_instance_predictions(instances)
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
        _save_score_file(results_dir, img_stem, img_path, 0, [], None)
        print("No cells detected. Drug efficacy score not computed.")
        return

    individual_scores = []

    for i in range(n_cells):
        mask = instances.pred_masks[i].numpy().astype("uint8") * 255
        box = instances.pred_boxes.tensor[i].numpy().astype(int)
        x1, y1, x2, y2 = box

        masked = cv2.bitwise_and(image, image, mask=mask)
        crop = masked[max(0, y1):y2, max(0, x1):x2]

        if crop.size == 0:
            continue

        hand = extract_handcrafted(crop)
        deep = extract_deep(model, device, image, box)
        feat = np.hstack([hand, deep]).reshape(1, -1)

        # Prediction & score mapping
        pred_class = xgb.predict(feat)[0]
        pred_score = REVERSE_MAP.get(pred_class, 0.0)

        individual_scores.append(pred_score)
        print(f"   - Cell {i + 1} class: {pred_class} | Score: {pred_score}")

    # ---- final image-level score ----
    if len(individual_scores) == 0:
        print("No valid cell features extracted. Drug efficacy score not computed.")
        _save_score_file(results_dir, img_stem, img_path, n_cells, [], None)
        return

    final_score = float(np.mean(individual_scores))

    _save_score_file(
        results_dir, img_stem, img_path,
        len(individual_scores), individual_scores, final_score,
    )

    print("-" * 40)
    print(f"Per-cell death scores : {individual_scores}")
    print(f"FINAL IMAGE SCORE     : {round(final_score, 3)}")
    print("-" * 40)


def _save_score_file(results_dir, img_stem, img_path, n_cells, scores, final_score):
    """Write a small text report to results/."""
    score_path = os.path.join(results_dir, f"{img_stem}_score.txt")
    with open(score_path, "w") as f:
        f.write(f"Input image: {img_path}\n")
        f.write(f"Number of cells detected: {n_cells}\n")
        if final_score is not None:
            f.write(f"Per-cell death scores: {scores}\n")
            f.write(f"Drug efficacy score (mean cell death): {final_score:.4f}\n")
        else:
            f.write("Drug efficacy score: N/A (no valid cells)\n")
    print(f"Score saved: {score_path}")


# ---------------------------------------------------------------------------
# Default test image
# ---------------------------------------------------------------------------
DEFAULT_IMAGE = os.path.join(SCRIPT_DIR, "img", "image_6h_4.jpg")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mask R-CNN + XGBoost inference -> segmentation image + drug efficacy score."
    )
    parser.add_argument(
        "--image",
        default=DEFAULT_IMAGE,
        help="Path to input microscopy image (default: img/image_6h_4.jpg)",
    )
    parser.add_argument(
        "--out_seg",
        default=None,
        help="Path for segmentation output image (default: results/<stem>_segmentation.png)",
    )
    args = parser.parse_args()
    main(args.image, args.out_seg)
