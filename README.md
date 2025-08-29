# Underwater-Bomb-Detection-using-CNN
This project uses Convolutional Neural Networks (CNN) with TensorFlow to detect underwater explosives from predefined image datasets.  
Code:
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(21, GPIO.OUT)
           
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model and labels
model = load_model("keras_model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    image = Image.open(image_path).convert("RGB")
    
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # turn the image into a numpy array
    image_array = np.asarray(image)
    
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Load the image into the array
    data[0] = normalized_image_array
    
    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    return class_name[2:], confidence_score

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            class_name, confidence = predict_image(filepath)
            
            # Prepare result message
            if class_name.strip() == "yes":
                GPIO.output(21, GPIO.HIGH)
                time.sleep(2)
                GPIO.output(21, GPIO.LOW)
                result = "Bomb Detected"
            else:
                result = "No Bomb Detected"
            
            return render_template('result.html', 
                                 image_path=filepath,
                                 result=result,
                                 confidence=float(confidence)*100)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
🚀 CNN-Based Underwater Bomb Detection (Flask + Raspberry Pi)

This project integrates a Convolutional Neural Network (CNN) with a Flask web application and Raspberry Pi GPIO to detect underwater explosives from images.

🔹 Features

Upload an image through a Flask web app.

Pre-trained CNN (keras_model.h5) classifies the image.

If a bomb is detected, the Raspberry Pi triggers a GPIO pin to activate external hardware (e.g., buzzer/LED).

Displays result and confidence score on a web page.

🔹 Tech Stack

Python, Flask (for web app)

Keras / TensorFlow (for CNN model)

PIL (Pillow) (for image preprocessing)

Raspberry Pi (RPi.GPIO) (for hardware alert system)

🔹 Workflow

User uploads an image (.jpg, .jpeg, .png).

Image is resized to 224x224 and normalized.

CNN model predicts whether the image contains a bomb.

If detected → Raspberry Pi GPIO pin 21 is activated for 2 seconds.

Results (Bomb Detected / No Bomb Detected + Confidence %) are displayed.

🔹 Example Output

Bomb Detected (Confidence: 96%) → Buzzer/LED activated

No Bomb Detected (Confidence: 88%) → Safe

🔹 How to Run
# Clone the repository
git clone https://github.com/yourusername/bomb-detection-cnn.git
cd bomb-detection-cnn

# Install dependencies
pip install flask keras pillow numpy RPi.GPIO

# Run the Flask app
python app.py


Then, open:

http://127.0.0.1:5000/
