from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the fine-tuned models
model_max = pickle.load(open("T_max_finetune.pkl", "rb"))
model_min = pickle.load(open("T_min_finetune.pkl", "rb"))

@app.route('/', methods=['GET'])
def hel():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect feature values from the form (24-hour input)
    present_tmax = [float(request.form.get(f'Present_Tmax_{i}', 0)) for i in range(2)]
    present_tmin = [float(request.form.get(f'Present_Tmin_{i}', 0)) for i in range(2)]

    # Store hourly predictions
    predictions_max = []
    predictions_min = []

    # Predict hourly maximum and minimum temperatures for the next 24 hours
    for i in range(2):
        features = np.array([[present_tmax[i], present_tmin[i]]])  # Using only two features for fine-tuned model
        prediction_max = model_max.predict(features)[0]
        prediction_min = model_min.predict(features)[0]
        
        predictions_max.append(prediction_max)
        predictions_min.append(prediction_min)

    # Calculate average temperatures for the next day
    avg_max_temp = np.mean(predictions_max)
    avg_min_temp = np.mean(predictions_min)

    # Render the result on a new HTML page
    return render_template('result.html', 
                           predictions_max=predictions_max,
                           predictions_min=predictions_min,
                           avg_max_temp=avg_max_temp,  # Pass as float for easier formatting in HTML
                           avg_min_temp=avg_min_temp)   # Pass as float for easier formatting in HTML

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
