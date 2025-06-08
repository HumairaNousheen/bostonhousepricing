import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np


app = Flask(__name__)



# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        json_data = request.get_json()
        data = json_data.get('data', json_data)  # support both formats
        print("Received Data:", data)
        input_array = np.array(list(data.values())).reshape(1, -1)
        new_data = scalar.transform(input_array)
        output = regmodel.predict(new_data)
        return jsonify({'prediction': output[0]})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
