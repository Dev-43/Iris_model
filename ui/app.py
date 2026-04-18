import streamlib as st
import requests

st.title("Iris Classifier")
sepal_length=st.number_input("Sepal Length")
sepal_width=st.number_input("Sepal Width")
petal_length=st.number_input("Petal Length")
petal_width=st.number_input("Petal Width")

if st.button("Predict"):
    data={
        "sepal_length":sepal_length,
        "sepal_width":sepal_width,
        "petal_length":petal_length,
        "petal_width":petal_width
    }

    res=requests.post("https://localhost::8000/predict",json=data)
    st.write(f"Predicted Class :{res.json['predicted_class']}")