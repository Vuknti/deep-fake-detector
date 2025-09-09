import cv2
import numpy as np
from app.utils import detect_faces, crop_face

def compute_features(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    (B, G, R) = cv2.split(face_crop.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    std_root = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
    mean_root = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
    colorfulness = std_root + (0.3 * mean_root)
    return {"blur": blur, "colorfulness": colorfulness}

def score_from_features(features):
    blur_score = 1.0 if features["blur"] < 100 else 0.0
    color_score = 1.0 if features["colorfulness"] > 40 else 0.0
    return (blur_score * 0.6 + color_score * 0.4)

def predict_image(image):
    boxes = detect_faces(image)
    if not boxes:
        return {"score": 0.5, "label": "unknown"}
    scores = []
    for box in boxes:
        crop = crop_face(image, box)
        feats = compute_features(crop)
        score = score_from_features(feats)
        scores.append(score)
    avg = sum(scores) / len(scores)
    return {"score": float(avg), "label": "fake" if avg > 0.5 else "real"}
