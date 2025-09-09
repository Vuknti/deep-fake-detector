import cv2

def extract_frames(video_path, n_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // n_frames, 1)
    frames = []
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) >= n_frames:
            break
    cap.release()
    return frames

# Haar cascade face detector (lightweight)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append((y, x+w, y+h, x))  # (top, right, bottom, left)
    return boxes

def crop_face(image, bbox):
    top, right, bottom, left = bbox
    return image[top:bottom, left:right]
