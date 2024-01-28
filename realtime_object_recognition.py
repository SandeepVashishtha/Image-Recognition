import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Function to preprocess an image
def preprocess_image(image):
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  return image

# Function to make predictions
def predict_image(image):
  image = preprocess_image(image)
  image = np.expand_dims(image, axis=0)
  predictions = model.predict(image)
  decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions.numpy())[0]
  return decoded_predictions

# Open a video capture stream
cap = cv2.VideoCapture(0)

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Check if the frame is successfully captured
  if not ret:
    print("Error: Failed to capture frame")
    break

  # Display the frame
  cv2.imshow('Real-time Object Recognition', frame)

  # Preprocess and make predictions
  predictions = predict_image(frame)

  # Print the top prediction
  print(predictions[0][1])

  # Break the loop if 'q' is pressed
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the video capture object
cap.release()

# Destroy all OpenCV windows
cv2.destroyAllWindows()
