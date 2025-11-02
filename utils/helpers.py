# utils/helpers.py

import cv2
import numpy as np
from keras.preprocessing.image import img_to_array

def preprocess_image(image):
    """Preprocess the image for emotion detection."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (48, 48))  # Resize to 48x48
    image = img_to_array(image)  # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image /= 255.0  # Normalize the image
    return image

def format_result(prediction):
    """Format the prediction result into a readable format."""
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    max_index = np.argmax(prediction[0])  # Get the index of the highest probability
    return emotions[max_index]  # Return the corresponding emotion label

def draw_label(image, label, position):
    """Draw the label on the image."""
    cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)