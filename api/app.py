from fastapi import FastAPI
import joblib
import sys,os

sys.path.append(os.path.abspath("../src"))
from schema import Iris

app=FastAPI()
model = joblib.load("../src/iris_model.pkl")
labels=['setosa','versicolor','virginaca']

@app.get("/")
def home():
    return{"message":"Welcome To the Iris Prediction Api"}

@app.post("/predict")
def predict(data : Iris):
    features =[[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    prediction=model.predict(features)[0]
    return {
        "class_id":int(prediction),
        "class_name":labels[prediction]
    }