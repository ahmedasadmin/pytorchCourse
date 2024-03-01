# export FLASK_APP=main.py
# export FLASK_ENV=development

from flask import Flask, request, jsonify
from pytorch_utils import transform_image, get_prediction 

app = Flask(__name__)
ALLOWED_EXTENSION = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSION
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({"error":"no file"})
        if not allowed_file(file.filename):
            return jsonify({"error":"not supported file"})
        try:
            img_bytes = file.read()
            tensor = transform_image(image_bytes=img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction':prediction.item(), 'class_name':str(prediction.item())}
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})

    return jsonify({"result" : 1})

if __name__ == "__main__":
    app.run()
 
