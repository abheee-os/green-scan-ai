import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Configure upload folder
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Load the trained model
MODEL_PATH = 'models/plant_disease_model.h5'
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_label(img_path):
    if model is None:
        return "Model not loaded"
    
    # Preprocessing the image
    # Note: Adjust target_size to match your model's input shape (160x160)
    i = image.load_img(img_path, target_size=(160, 160))
    i = image.img_to_array(i)
    i = i / 255.0  # Normalize if the model expects normalized input
    i = i.reshape(1, 160, 160, 3)
    
    p = model.predict(i)
    
    # Class names provided by the user (Corrected Order)
    class_names = [
        'Apple Scab', 
        'Apple Black Rot', 
        'Apple Cedar Apple Rust', 
        'Apple Healthy', 
        'Blueberry Healthy', 
        'Cherry (including sour) Powdery Mildew', 
        'Cherry (including sour) Healthy', 
        'Corn (maize) Cercospora Leaf Spot/Gray Leaf Spot', 
        'Corn (maize) Common Rust', 
        'Corn (maize) Northern Leaf Blight', 
        'Corn (maize) Healthy'
    ]
    
    predicted_class_index = np.argmax(p)
    confidence = np.max(p)
    
    # Check if the index is within the range of our class names
    if 0 <= predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
        return f"{predicted_class} (Confidence: {confidence*100:.2f}%)"
    else:
        return f"Unknown Class Index: {predicted_class_index} (Confidence: {confidence*100:.2f}%)"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                # Fallback if filename becomes empty
                if not filename:
                    filename = 'uploaded_image.jpg'
                    
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Make prediction
                prediction = predict_label(file_path)
                
                return render_template('result.html', prediction=prediction, image_file=filename)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"An error occurred while processing the image: {str(e)}", 500

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)

