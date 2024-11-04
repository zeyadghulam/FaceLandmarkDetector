#################### This code enables webcams in Google Colab:

# Import bncessary plug-ins:
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

# Create a function called 'capture_image' to obtain webcam input:
def capture_image(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);
      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});
      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();
      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
    
  # The photo will be saved in Google Colab as 'photo.jpg':  
  return filename



#################### This code uses the webcam image to find your skeleton:

# Import necessary plug-ins:
import mediapipe as mp
import cv2
from google.colab.patches import cv2_imshow

# Renaming some mediapipe functionalities to short-hand form:
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Create a function called 'find_skeleton' that uses the webcam image:
def find_skeleton():
  
  # Load in the image:
  cap = cv2.VideoCapture('photo.jpg')
  width, height  = cap.get(3), cap.get(4)

  # Use mediapipe pipeline to process the image:
  with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference:
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw landmark annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles
        .get_default_face_mesh_contours_style())
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())

    # Show the image in the web browser:
    cv2_imshow(image)

    # Free up software and hardware resources:
    cap.release()