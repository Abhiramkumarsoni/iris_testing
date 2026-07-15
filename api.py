# let's create a api using flask to serve the model
from flask import Flask, request, jsonify
# joblib is used to load the model that we have saved in the dist
import numpy as np
import joblib

app=Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Iris Prediction API"

#load the model and create a predict endpoint

model = joblib.load("random_forest_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Convert the input data to a numpy array
    input_data = np.array(list(data.values())).reshape(1, -1)
    # Make a prediction
    prediction = model.predict(input_data)
    return jsonify({"prediction": int(prediction[0])})


# Run the app
if __name__=="__main__":
    app.run(debug=True,host='0.0.0.0',port=5000)
