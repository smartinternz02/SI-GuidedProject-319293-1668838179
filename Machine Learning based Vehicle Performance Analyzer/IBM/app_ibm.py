import flask
from flask import request, render_template,jsonify
from flask_cors import CORS
import joblib
import json
import requests

API_KEY = "fpsbvCw3CFhouF_CC10k6JpZJW34Q9ozTst1e4AqHU9g"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

app = flask.Flask(__name__, static_url_path='')
CORS(app)

@app.route('/', methods=['GET'])
def sendHomePage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predictPerformance():
    cylinders = float(request.form['cylinders'])
    displacement = float(request.form['displacement'])
    horsepower = float(request.form['horsepower'])
    weight = float(request.form['weight'])
    acceleration = float(request.form['acceleration'])
    modelyear = float(request.form['modelyear'])
    origin = float(request.form['origin'])

    X = [[cylinders,displacement,horsepower,
          weight,acceleration,modelyear,origin]]



    payload_scoring = {"input_data": [{"field": [['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration','modelyear','origin']],
                                       "values": X}]}

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/95813d6f-522d-4811-bb50-df4da823df20/predictions?version=2022-04-18', json=payload_scoring, headers={'Authorization': 'Bearer ' + mltoken})

    

    response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/95813d6f-522d-4811-bb50-df4da823df20/predictions?version=2022-04-18', json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    print("Scoring response")
    print(response_scoring.json())
    pred = response_scoring.json()


    output = pred['predictions'][0]['values'][0][0]
    print(output)
    
    return render_template('predict.html', predict=output)


if __name__ == "__main__":
    app.run(debug=False)