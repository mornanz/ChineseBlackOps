import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"
SOURCE = 0

# Ładujemy model emocji
model = YOLO(MODEL_PATH)

# Haar cascade (do wykrywania twarzy)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Wykrywamy twarze
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Wycinamy twarz
        face_img = frame[y:y+h, x:x+w]

        # 2. Przepuszczamy przez YOLO (emocje)
        results = model.predict(face_img, conf=0.25, verbose=False)
        result = results[0]

        if len(result.boxes) > 0:
            cls_id = int(result.boxes[0].cls[0])
            label = model.names[cls_id]
            conf = float(result.boxes[0].conf[0])

            # Rysujemy bounding box + etykietę
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        else:
            # Sam box twarzy
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Emocje - YOLO + FaceDetector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
