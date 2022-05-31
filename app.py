
   
from flask import Flask, jsonify, request
from Classifier import getPrediction

app = Flask(__name__)
@app.route('/predict-alpha', methods=['POST'])

def predict_alpha():
    image = request.files.get('digit')
    prediction = getPrediction(image)
    return jsonify({
        "prediction":prediction
        }), 200

if __name__ == '__main__':
    app.run(debug=True)