from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import numpy as np


app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("model.keras")
print("Model loaded successfully!")

# Define the disease classes
class_indices = {
    'Apple___Apple_scab': 0, 'Apple___Black_rot': 1, 'Apple___Cedar_apple_rust': 2, 
    'Apple___healthy': 3, 'Blueberry___healthy': 4, 'Cherry_(including_sour)___Powdery_mildew': 5, 
    'Cherry_(including_sour)___healthy': 6, 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 7, 
    'Corn_(maize)___Common_rust_': 8, 'Corn_(maize)___Northern_Leaf_Blight': 9, 'Corn_(maize)___healthy': 10, 
    'Grape___Black_rot': 11, 'Grape___Esca_(Black_Measles)': 12, 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 13, 
    'Grape___healthy': 14, 'Orange___Haunglongbing_(Citrus_greening)': 15, 'Peach___Bacterial_spot': 16, 
    'Peach___healthy': 17, 'Pepper,_bell___Bacterial_spot': 18, 'Pepper,_bell___healthy': 19, 
    'Potato___Early_blight': 20, 'Potato___Late_blight': 21, 'Potato___healthy': 22, 'Raspberry___healthy': 23, 
    'Soybean___healthy': 24, 'Squash___Powdery_mildew': 25, 'Strawberry___Leaf_scorch': 26, 
    'Strawberry___healthy': 27, 'Tomato___Bacterial_spot': 28, 'Tomato___Early_blight': 29, 
    'Tomato___Late_blight': 30, 'Tomato___Leaf_Mold': 31, 'Tomato___Septoria_leaf_spot': 32, 
    'Tomato___Spider_mites Two-spotted_spider_mite': 33, 'Tomato___Target_Spot': 34, 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 35, 'Tomato___Tomato_mosaic_virus': 36, 'Tomato___healthy': 37
}
class_map = {value: key for key, value in class_indices.items()}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files or not request.files['file']:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        try:
            img = Image.open(file)
        except UnidentifiedImageError:
            return jsonify({'error': 'Invalid image file. Ensure the file is a valid image.'}), 400

        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        try:
            prediction = model.predict(img_array)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(np.max(prediction))

            if predicted_class_idx in class_map:
                predicted_class_name = class_map[predicted_class_idx]
                return jsonify({
                    'predicted_class': predicted_class_name,
                    'confidence': confidence
                })
            else:
                return jsonify({'error': 'Disease not supported yet'}), 400

        except Exception as e:
            return jsonify({'error': f'Prediction failed: {e}'}), 500

    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'}), 500

# Error handling for 404
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found. The API endpoint you are trying to access does not exist.'}), 404

if __name__ == '__main__':
    app.run(debug=True)
