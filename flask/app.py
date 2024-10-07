import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open(r'D:\Diabetes-Prediction-master\flask\knn_model.pkl', 'rb'))

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        print("Received input features:", float_features)
        final_features = [np.array(float_features)]
        print("Final features for prediction:", final_features)

        prediction = model.predict(sc.transform(final_features))

        if prediction == 1:
            pred = "You may have Diabetes, please consult a Doctor."
        elif prediction == 0:
            pred = "You don't have Diabetes."

        return render_template('index.html', prediction_text='{}'.format(pred))

    except Exception as e:
        print("Error during prediction:", e)
        return render_template('index.html', prediction_text='Error occurred during prediction.')

if __name__ == "__main__":
    app.run(debug=True)
