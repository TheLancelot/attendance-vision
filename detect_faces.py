from argparse import ArgumentParser
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import face_recognition
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = BASE_DIR / "photos/class_photo_2.jpeg"
DEFAULT_OUTPUT = BASE_DIR / "detections/detected_faces.jpg"
MODEL_PATH = BASE_DIR / "yolov8n-face.pt"
MODEL_URL = (
  "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/"
  "yolov8n-face.pt"
)


def ensure_model():
  if not MODEL_PATH.exists():
    print(f"Downloading {MODEL_PATH.name}...")
    urlretrieve(MODEL_URL, MODEL_PATH)
  return YOLO(str(MODEL_PATH))


def yolo_face_locations(image_bgr, model):
  results = model.predict(
    source=image_bgr,
    conf=0.25,
    imgsz=1280,
    max_det=1000,
    verbose=False,
  )
  boxes = results[0].boxes.xyxy.cpu().numpy()
  return [
    (int(top), int(right), int(bottom), int(left))
    for left, top, right, bottom in boxes
  ]


def draw_boxes(image_bgr, face_locations, color):
  for top, right, bottom, left in face_locations:
    cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 3)


def main():
  parser = ArgumentParser(description="Compare YOLOv8-face and face_recognition.")
  parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
  parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
  args = parser.parse_args()

  if not args.image.exists():
    raise FileNotFoundError(f"Input image not found: {args.image}")

  image_rgb = face_recognition.load_image_file(args.image)
  image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

  face_recognition_locations = face_recognition.face_locations(
    image_rgb,
    number_of_times_to_upsample=2,
    model="hog",
  )
  yolo_locations = yolo_face_locations(image_bgr, ensure_model())

  if len(yolo_locations) > len(face_recognition_locations):
    selected_method = "YOLOv8n-face"
    selected_locations = yolo_locations
    selected_color = (0, 255, 0)
  else:
    selected_method = "face_recognition"
    selected_locations = face_recognition_locations
    selected_color = (0, 0, 255)

  draw_boxes(image_bgr, selected_locations, selected_color)
  cv2.imwrite(str(args.output), image_bgr)
  print(f"YOLOv8n-face detections: {len(yolo_locations)}")
  print(f"face_recognition detections: {len(face_recognition_locations)}")
  print(f"Displayed {len(selected_locations)} boxes from {selected_method}.")
  print(f"Saved result to {args.output}")

  # cv2.imshow("Face detections", image_bgr)
  cv2.waitKey(0)
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
