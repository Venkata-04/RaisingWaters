# Rising Waters — Flask App

## Before running

Place your two saved files from the model-building notebook directly in this folder
(same level as app.py):

- floods.save      (the trained Gradient Boosting / XGBoost model)
- transform.save   (the fitted StandardScaler)

## Install dependencies

    pip3 install flask joblib scikit-learn numpy

## Run

    python3 app.py

Then open http://127.0.0.1:5000 in your browser.

## Routes

- /         Home page, enter weather readings
- /predict  Handles the form submission, runs the model, shows the result
- /history  Shows every prediction made this session
