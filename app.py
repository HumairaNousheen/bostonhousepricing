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
        data = request.json['data']
        input_array = np.array(list(data.values())).reshape(1, -1)
        new_data = scalar.transform(input_array)
        output = regmodel.predict(new_data)
        return jsonify(output[0])
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Convert form inputs to floats safely
        data = []
        for val in request.form.values():
            data.append(float(val))
        
        # Scale and predict
        final_input = scalar.transform(np.array(data).reshape(1, -1))
        output = regmodel.predict(final_input)[0]

        # Pass prediction and original inputs back to the template
        input_dict = dict(zip(request.form.keys(), data))
        return render_template(
            "home.html", 
            prediction_text=f"Predicted House Price: ${round(output * 1000, 2)}",
            inputs=input_dict
        )
    except Exception as e:
        return render_template(
            "home.html", 
            prediction_text=f"Error: {e}."
        )

if __name__ == "__main__":
    app.run(debug=True)
