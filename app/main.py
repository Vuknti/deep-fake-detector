from fastapi import FastAPI, UploadFile, File
from app.ml.detector import predict_image
from app.utils import extract_frames
import cv2, tempfile, os

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Deepfake Detector API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    result = {}
    if suffix.lower() in [".jpg", ".jpeg", ".png"]:
        img = cv2.imread(tmp_path)
        result = predict_image(img)
    elif suffix.lower() in [".mp4", ".avi", ".mov"]:
        frames = extract_frames(tmp_path, n_frames=8)
        scores = [predict_image(f)["score"] for f in frames]
        avg_score = sum(scores) / len(scores)
        result = {"score": avg_score, "label": "fake" if avg_score > 0.5 else "real"}
    else:
        result = {"error": "Unsupported file format"}

    os.remove(tmp_path)
    return result
