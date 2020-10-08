import pickle
import os
import pandas as pd
from mytoolbox.mytoolbox import *
from flask import Flask, request

# Load model
model = pickle.load(open('models/lgbm_pipe_tunning.pkl', 'rb'))


# Instanciate Flask
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    pp = PreProcessingTransformer()
    test_json = request.get_json()

    # Collect Data
    if test_json:
        if isinstance(test_json, dict):
            df_raw = pd.DataFrame(test_json, index=[0])
            df_raw_processed = pp.fit_transform(df_raw)
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            df_raw_processed = pp.fit_transform(df_raw)

    # Predictions
    pred = model.predict(df_raw_processed)

    df_raw_processed['prediction'] = pred

    return df_raw_processed.to_json(orient='records')


@app.route('/predict_thresh', methods=['POST'])
def predict_thresh():

    pp = PreProcessingTransformer()
    test_json = request.get_json()

    # Collect Data
    if test_json:
        if isinstance(test_json, dict):
            df_raw = pd.DataFrame(test_json, index=[0])
            df_raw_processed = pp.fit_transform(df_raw)
        else:
            df_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            df_raw_processed = pp.fit_transform(df_raw)

    # Predictions
    pred_prob = model.predict_proba(df_raw_processed)

    for i in range(0, len(pred_prob)):

        if pred_prob[i, 1] >= 0.4:
            pred_prob[i, 1] = 1
        else:
            pred_prob[i, 1] = 0

    df_raw_processed['prediction'] = pred_prob[:, 1]

    return df_raw_processed.to_json(orient='records')


if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)