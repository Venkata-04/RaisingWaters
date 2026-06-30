from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and the fitted scaler.
# These were saved at the end of Epic 4 (Model Building).
model = joblib.load('floods.save')
scaler = joblib.load('transform.save')

# In-memory store of predictions made during this run.
# This resets whenever the server restarts -- it is not a database.
prediction_history = []


@app.route('/')
def home():
    """Renders the input form (Home Page)."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Reads form input, scales it, runs the model, and shows the result."""

    # Pull values from the submitted form, in the exact column order
    # the model was trained on: Temp, Humidity, Cloud Cover, ANNUAL,
    # Jan-Feb, Mar-May, Jun-Sep, Oct-Dec, avgjune, sub
    temp = float(request.form['temp'])
    humidity = float(request.form['humidity'])
    cloud_cover = float(request.form['cloud_cover'])
    annual = float(request.form['annual'])
    jan_feb = float(request.form['jan_feb'])
    mar_may = float(request.form['mar_may'])
    jun_sep = float(request.form['jun_sep'])
    oct_dec = float(request.form['oct_dec'])
    avgjune = float(request.form['avgjune'])
    sub = float(request.form['sub'])

    features = np.array([[temp, humidity, cloud_cover, annual,
                           jan_feb, mar_may, jun_sep, oct_dec,
                           avgjune, sub]])

    # Scale using the SAME scaler fitted during training, then predict.
    scaled_features = scaler.transform(features)
    prediction = int(model.predict(scaled_features)[0])

    inputs = {
        'temp': temp,
        'humidity': humidity,
        'cloud_cover': cloud_cover,
        'annual': annual,
        'jun_sep': jun_sep,
    }

    # Log this prediction for the History page.
    record = dict(inputs)
    record['prediction'] = prediction
    prediction_history.append(record)

    return render_template('result.html', prediction=prediction, inputs=inputs)


@app.route('/history')
def history():
    """Shows every prediction made so far in this session."""
    return render_template('history.html', records=prediction_history)


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
