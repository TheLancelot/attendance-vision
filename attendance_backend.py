from pathlib import Path

import cv2
import face_recognition
import numpy as np

from detect_faces import ensure_model, yolo_face_locations
import register_students as registry

BASE_DIR = Path(__file__).resolve().parent
STUDENTS_DIR = BASE_DIR / "students"
TEST_DIR = BASE_DIR / "photos"
DETECTIONS_DIR = BASE_DIR / "detections"

# inference
def process_classroom_attendance(classroom_image_path, output_image_path=None):
    """
    Detects all faces in a group photo, generates embeddings for each,
    and matches them against the registered students database.
    """
    print(f"Processing classroom photo: {classroom_image_path}")
    
    classroom_image = face_recognition.load_image_file(classroom_image_path)
    
    classroom_image_bgr = cv2.cvtColor(classroom_image, cv2.COLOR_RGB2BGR)
    # face_recognition_locations = face_recognition.face_locations(
    #     img=classroom_image,
    #     number_of_times_to_upsample=1,
    #     model="hog",
    # )
    
    yolo_locations = yolo_face_locations(
        classroom_image_bgr,
        ensure_model(),
    )

    # if len(yolo_locations) > len(face_recognition_locations):
    #     face_locations = yolo_locations
    #     selected_detector = "YOLOv8n-face"
    # else:
    #     face_locations = face_recognition_locations
    #     selected_detector = "face_recognition"

    face_locations=yolo_locations
    selected_detector="YOLOv8n-face"

    print(f"YOLOv8n-face detections: {len(yolo_locations)}")
    # print(f"face_recognition detections: {len(face_recognition_locations)}")
    print(f"Using {len(face_locations)} locations from {selected_detector}")
    
    face_embeddings = face_recognition.face_encodings(classroom_image, face_locations,num_jitters=10,model="large")

    print(f"Detected {len(face_embeddings)} face(s) in the classroom photo.")
    present_students = set()
    annotated_image = classroom_image_bgr

    # Loop through each face found in the classroom photo
    for face_location, current_face_embedding in zip(face_locations, face_embeddings):
        # Compare current face against ALL registered student faces tolerance=0.6 is the default (lower is stricter, higher is looser)
        face_distances = face_recognition.face_distance(
            registry.known_face_embeddings,
            current_face_embedding,
        )
        print(face_distances)
        print(np.min(face_distances))

        name = "Unknown"
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] <= 0.6:
                name = registry.known_face_names[best_match_index]
                present_students.add(name)
                
        top, right, bottom, left = face_location
        box_color = (0, 180, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(
            annotated_image,
            (left, top),
            (right, bottom),
            box_color,
            2,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            name,
            font,
            font_scale,
            font_thickness,
        )
        label_top = max(0, top - text_height - baseline - 6)
        label_bottom = top if label_top > 0 else min(
            annotated_image.shape[0],
            top + text_height + baseline + 6,
        )

        cv2.rectangle(
            annotated_image,
            (left, label_top),
            (left + text_width + 8, label_bottom),
            box_color,
            cv2.FILLED,
        )
        text_y = label_bottom - baseline - 3
        cv2.putText(
            annotated_image,
            name,
            (left + 4, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )
        
        print(f" - Found face matching: {name}")

    if output_image_path is None:
        output_image_path = BASE_DIR / "attendance_result.jpg"

    output_image_path = Path(output_image_path)
    cv2.imwrite(str(output_image_path), annotated_image)
    print(f"Annotated attendance image saved to: {output_image_path}")

    return list(present_students)

if __name__ == "__main__":

    STUDENTS_DIR.mkdir(exist_ok=True)
    
    # load cached embedding/build
    registry.register_students_from_folder(str(STUDENTS_DIR))

    test_group_photo = TEST_DIR / "team_photo.jpg"
    output_image = DETECTIONS_DIR / "team_photo.jpg"
    
    if test_group_photo.exists():
        attendance_sheet = process_classroom_attendance(
            str(test_group_photo),
            str(output_image),
        )
        print("\n--- FINAL ATTENDANCE SHEET ---")
        for student in attendance_sheet:
            print(f"[PRESENT] {student}")
    else:
        print(f"\nTo test the inference, place a group photo named '{test_group_photo}' in this directory.")
