from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_qr_data(image_path):
    try:
        # Read the image
        img = cv2.imread(image_path)
        
        # Decode QR codes
        decoded_objects = decode(img)
        
        if decoded_objects:
            results = []
            for obj in decoded_objects:
                results.append({
                    'data': obj.data.decode('utf-8'),
                    'type': obj.type,
                    'rect': {
                        'left': obj.rect.left,
                        'top': obj.rect.top,
                        'width': obj.rect.width,
                        'height': obj.rect.height
                    }
                })
            return results
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

@app.route('/scan', methods=['POST'])
def scan_qr():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        qr_data = extract_qr_data(filepath)
        
        # Clean up
        os.remove(filepath)
        
        if qr_data:
            return jsonify({'results': qr_data})
        else:
            return jsonify({'error': 'No QR code found'}), 404
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)