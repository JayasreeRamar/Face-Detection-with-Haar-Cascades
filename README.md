# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows  

### PROGRAM:
## NAME: JAYASREE R
## REG NO: 212223230087
```
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

withglass = cv2.imread('image_02.png', 0)
group = cv2.imread('img_3.jpg', 0)

plt.imshow(withglass, cmap='gray')
plt.title("With Glasses")
plt.show()

plt.imshow(group, cmap='gray')
plt.title("Group Image")
plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty():
    raise IOError("Error loading face cascade XML file")
if eye_cascade.empty():
    raise IOError("Error loading eye cascade XML file")

def detect_face(img, scaleFactor=1.1, minNeighbors=5):
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in face_rects:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

def detect_eyes(img):
    face_img = img.copy()
    eyes = eye_cascade.detectMultiScale(face_img)
    for (x, y, w, h) in eyes:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 2)
    return face_img

result_withglass_faces = detect_face(withglass)
plt.imshow(result_withglass_faces, cmap='gray')
plt.title("Faces in With Glasses Image")
plt.show()

result_group_faces = detect_face(group)
plt.imshow(result_group_faces, cmap='gray')
plt.title("Faces in Group Image")
plt.show()

plt.imshow(result_withglass_eyes_refined, cmap='gray')
plt.title("Refined Eye Detection - With Glasses Image")
plt.show()

plt.imshow(result_group_eyes_refined_tuned_v2, cmap='gray')
plt.title("Further Tuned Eye Detection - Group Image (V2)")
plt.show()
```
### OUTPUT :
<img width="565" height="548" alt="Screenshot 2025-11-15 105914" src="https://github.com/user-attachments/assets/b7941cf5-798f-4f1c-8188-5b30cb88b022" />
<img width="695" height="492" alt="Screenshot 2025-11-15 105931" src="https://github.com/user-attachments/assets/8ce7e7e4-90aa-4b4d-ba61-594648f86642" />
<img width="573" height="554" alt="Screenshot 2025-11-15 105950" src="https://github.com/user-attachments/assets/3d62a04d-2785-4a83-aea4-7d1c0e946b48" />
<img width="702" height="490" alt="Screenshot 2025-11-15 110004" src="https://github.com/user-attachments/assets/76b76967-de43-48de-989c-bdc3c28d9c68" />
<img width="574" height="562" alt="Screenshot 2025-11-15 110019" src="https://github.com/user-attachments/assets/3b780370-13a3-46c0-ac66-2ae800735928" />
<img width="698" height="505" alt="Screenshot 2025-11-15 110033" src="https://github.com/user-attachments/assets/19d7300a-eec6-475e-b542-11d395a38a04" />

### RESULT:
Thus the program to implement Face Detection using Haar Cascades was executed successfully.
