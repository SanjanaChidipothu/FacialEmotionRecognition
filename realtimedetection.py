import cv2 # type: ignore
from keras.models import model_from_json # type: ignore
import numpy as np # type: ignore

# Load the pre-trained model architecture and weights
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load the pre-trained face cascade classifier
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the face image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Dictionary to map predicted labels to emotions
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Open webcam
webcam = cv2.VideoCapture(1)

# Main loop for real-time detection
while True:
    ret, frame = webcam.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    try:
        # Iterate over detected faces
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = gray[y:y+h, x:x+w]
            # Resize to match model input shape
            resized_face = cv2.resize(face_img, (48, 48))
            # Extract features and normalize
            img = extract_features(resized_face)
            # Make prediction
            pred = model.predict(img)
            # Get predicted label
            prediction_label = labels[pred.argmax()]
            # Display prediction label on the frame
            cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the frame
        cv2.imshow('Face Emotion Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
