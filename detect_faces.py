import cv2
import face_recognition
from ultralytics import YOLO

image_path = "IMG_Party.jpg"
model_path = (
  "https://github.com/YapaLab/yolo-face/releases/download/"
  "1.0.0/yolov8n-face.pt"
)

image_bgr = cv2.imread(image_path)
if image_bgr is None:
  raise FileNotFoundError(f"Could not load image: {image_path}")

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
hog_locations = face_recognition.face_locations(
  image_rgb,
  number_of_times_to_upsample=1,
  model="hog",
)
hog_boxes = [
  (left, top, right, bottom)
  for top, right, bottom, left in hog_locations
]

model = YOLO(model_path)
results = model(image_bgr, conf=0.25, imgsz=1280, verbose=False)
yolo_boxes = [
  tuple(box)
  for box in results[0].boxes.xyxy.cpu().numpy().astype(int)
]


def box_iou(first_box, second_box):
  first_left, first_top, first_right, first_bottom = first_box
  second_left, second_top, second_right, second_bottom = second_box
  intersection_width = max(
    0,
    min(first_right, second_right) - max(first_left, second_left),
  )
  intersection_height = max(
    0,
    min(first_bottom, second_bottom) - max(first_top, second_top),
  )
  intersection_area = intersection_width * intersection_height
  first_area = max(0, first_right - first_left) * max(0, first_bottom - first_top)
  second_area = max(0, second_right - second_left) * max(0, second_bottom - second_top)
  union_area = first_area + second_area - intersection_area
  return intersection_area / union_area if union_area else 0


combined_boxes = list(yolo_boxes)
for hog_box in hog_boxes:
  if not any(box_iou(hog_box, yolo_box) >= 0.3 for yolo_box in combined_boxes):
    combined_boxes.append(hog_box)


def annotate(image, boxes, color):
  annotated_image = image.copy()
  for left, top, right, bottom in boxes:
    cv2.rectangle(
      annotated_image,
      (left, top),
      (right, bottom),
      color,
      3,
    )
  return annotated_image

cv2.imwrite("detected_faces_hog.jpg", annotate(image_bgr, hog_boxes, (0, 255, 0)))
cv2.imwrite("detected_faces_yolo.jpg", annotate(image_bgr, yolo_boxes, (0, 0, 255)))
cv2.imwrite("detected_faces_combined.jpg", annotate(image_bgr, combined_boxes, (255, 0, 0)))

print(f"HOG detections: {len(hog_boxes)}")
print(f"YOLO detections: {len(yolo_boxes)}")
print(f"Combined detections: {len(combined_boxes)}")
