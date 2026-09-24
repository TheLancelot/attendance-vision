import os
import json
import cv2
import face_recognition
import numpy as np

known_face_embeddings = []
known_face_names = []

CACHE_FILE = "students_embeddings.npz"
CACHE_META_FILE = "students_embeddings_meta.json"


def _student_files_signature(folder_path):
    """Creates a signature based on filenames, sizes, and modification times."""
    files = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, filename)
            stat = os.stat(path)
            files.append({
                "filename": filename,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns
            })

    return files


def load_cached_embeddings(folder_path="students"):
    """Loads cached embeddings if the student images have not changed."""
    if not os.path.exists(CACHE_FILE) or not os.path.exists(CACHE_META_FILE):
        return False

    current_signature = _student_files_signature(folder_path)

    with open(CACHE_META_FILE, "r", encoding="utf-8") as file:
        cached_signature = json.load(file)

    if current_signature != cached_signature:
        return False

    data = np.load(CACHE_FILE, allow_pickle=False)

    known_face_embeddings.clear()
    known_face_names.clear()

    known_face_embeddings.extend(data["embeddings"])
    known_face_names.extend(data["names"].tolist())

    print(f"Loaded {len(known_face_names)} student embeddings from cache.")
    return True


def register_students_from_folder(folder_path="students"):
    """Generates and caches embeddings for student images."""
    global known_face_embeddings, known_face_names

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    if load_cached_embeddings(folder_path):
        return

    print("--- Starting Student Registration ---")

    embeddings = []
    names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            student_name = os.path.splitext(filename)[0].replace("_", " ")
            image_path = os.path.join(folder_path, filename)

            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(face_image=image,num_jitters=100,model="large")

            if face_encodings:
                embeddings.append(face_encodings[0])
                names.append(student_name)
                print(f"Successfully registered: {student_name}")
            else:
                print(f"Skipping {filename}: No face detected.")

    known_face_embeddings = embeddings
    known_face_names = names

    np.savez(
        CACHE_FILE,
        embeddings=np.asarray(embeddings),
        names=np.asarray(names)
    )

    with open(CACHE_META_FILE, "w", encoding="utf-8") as file:
        json.dump(_student_files_signature(folder_path), file, indent=2)

    print("--- Registration Complete; embeddings cached ---\n")