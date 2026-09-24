import cv2
import face_recognition

# 1. Load the image file
image_path = 'class_photo_2.jpeg'
image = face_recognition.load_image_file(image_path)

# 2. Find all face locations in the image (using default HOG model)
face_locations = face_recognition.face_locations(image)
print(len(face_locations))

# 3. Convert the image from RGB (face_recognition) to BGR (OpenCV)
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# 4. Loop through each face found and draw a bounding box
for top, right, bottom, left in face_locations:
  # Draw a rectangle around the face (Color: Green, Thickness: 2)
  cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 0, 255), 3)

# 5. Display the resulting image
cv2.imwrite('detected_faces.jpg', image_bgr)
# cv2.imshow('Detected Faces', image_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
