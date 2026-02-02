import cv2
import numpy as np
import torch
import joblib
import argparse

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis


# -------------------------------
# Handcrafted features
# -------------------------------
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []

    # GLCM
    glcm = graycomatrix(gray,[1],[0],256,True,True)
    for p in ['energy','contrast','correlation','homogeneity','ASM','dissimilarity']:
        feats.append(graycoprops(glcm,p)[0,0])

    # Histogram stats
    pix = gray.flatten()
    feats += [pix.mean(), pix.std(), skew(pix), kurtosis(pix)]

    # LBP
    lbp = local_binary_pattern(gray,8,1,"uniform")
    hist,_ = np.histogram(lbp.ravel(),bins=np.arange(0,12))
    hist = hist/(hist.sum()+1e-6)
    feats += list(hist)

    return np.array(feats)


# -------------------------------
# Setup Detectron2
# -------------------------------
def load_segmenter(weights):
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


# -------------------------------
# Main
# -------------------------------
def main(img_path):

    predictor = load_segmenter("weights/mask_rcnn.pth")
    xgb = joblib.load("weights/xgboost.pkl")

    img = cv2.imread(img_path)
    outputs = predictor(img)

    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()

    scores = []

    for b in boxes:
        x1,y1,x2,y2 = b.astype(int)
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        feats = extract_features(crop).reshape(1,-1)
        pred = xgb.predict(feats)[0]
        scores.append(pred)

    if len(scores)==0:
        print("No cells detected.")
        return

    print("\nCell scores:", scores)
    print("Average cell death score:", np.mean(scores))


# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    main(args.image)
