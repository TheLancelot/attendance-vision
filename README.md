# Attendance Vision

Attendance Vision is a Python computer-vision project that marks student attendance from a classroom group photo. It uses the `face_recognition` library to create face embeddings for registered students, compares detected faces against those embeddings, writes the recognized names above bounding boxes, and saves an annotated image.

## Features

- Registers student face images from the `students/` directory.
- Converts each registered face into a 128-dimensional face embedding.
- Caches the embeddings in `students_embeddings.npz`.
- Stores file metadata in `students_embeddings_meta.json` to detect changes.
- Automatically rebuilds the cache when a student image is added, removed, renamed, or modified.
- Detects multiple faces in a classroom image.
- Matches faces using Euclidean face distance and a tolerance of `0.6`.
- Draws a bounding box and student name for each detected face.
- Labels unmatched faces as `Unknown`.
- Saves the annotated result as `attendance_result.jpg`.

## Project Structure

```text
attendance_vision/
|-- attendance_backend.py          # Main attendance and annotation pipeline
|-- detect_faces.py                # Simple face detection example
|-- register_students.py           # Registration and embedding cache logic
|-- students/                      # Registered student images
|-- students_embeddings.npz       # Cached face embeddings
|-- students_embeddings_meta.json  # Cache validation metadata
|-- attendance_result.jpg          # Generated annotated image
|-- README.md
```

## Requirements

- Python 3.8 or newer
- OpenCV
- NumPy
- `face_recognition`
- `dlib` (installed as a dependency of `face_recognition`)

Install the Python packages with:

```powershell
pip install opencv-python numpy face_recognition
```

On Windows, installing `dlib` may require a compatible prebuilt package or a C++ build environment, depending on the Python version.

## Register Student Images

Place one clear image per student inside `students/`.

The filename becomes the student's displayed name. For example:

```text
students/
|-- Alice_Johnson.jpg
|-- Rahul_Sharma.png
|-- Maria_Garcia.jpeg
```

These names are displayed as:

```text
Alice Johnson
Rahul Sharma
Maria Garcia
```

Each image should contain one clearly visible face. Images without a detectable face are skipped.

## Run Attendance

Place the classroom group photo in the project directory with the name:

```text
class_photo_2.jpeg
```

Run the main script from the project directory:

```powershell
cd D:\college_coding\attendance_vision
python attendance_backend.py
```

The script will:

1. Load the existing cached student embeddings if they are still valid.
2. Rebuild the embeddings if the student image files have changed.
3. Detect faces in `class_photo_2.jpeg`.
4. Match detected faces against the registered students.
5. Print the present students in the terminal.
6. Save the annotated image as `attendance_result.jpg`.

## Cache Behavior

The first run creates the embedding cache. Later runs reuse it, so the registration step does not repeat unnecessarily.

The cache is rebuilt automatically when the contents of `students/` change. The cache currently checks each image's:

- Filename
- File size
- Last modification time

To force a rebuild manually, delete these files and run the script again:

```text
students_embeddings.npz
students_embeddings_meta.json
```

## Other Script

`detect_faces.py` is a smaller example that detects faces and saves bounding boxes to `detected_faces.jpg`. It does not perform student recognition or attendance matching.

## Limitations

- Recognition quality depends on image quality, lighting, pose, and face visibility.
- The current registration flow uses the first detected face in each student image.
- A single reference image per student may be less reliable than multiple reference images.
- The attendance result is saved as an image; it is not currently exported to a CSV or database.
- The matching tolerance may need adjustment for a particular dataset.

## Paste-Ready Description

Attendance Vision is a face-recognition-based classroom attendance system built with Python, OpenCV, NumPy, and the `face_recognition` library. The project registers student images as face embeddings, caches those embeddings to avoid repeating expensive processing, detects faces in a classroom group photo, and compares them with the registered database. Recognized students are labeled above their bounding boxes, unknown faces are marked accordingly, and the final annotated attendance image is saved for review.
