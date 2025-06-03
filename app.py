import streamlit as st
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

try:
    model = joblib.load("./models/decision_tree_classifier_default_42.sav")
    st.write("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    raise

class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

def main():
    st.title("Iris Flower Classifier")

    # Input fields
    st.header("Enter Feature Values")
    val1 = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    val2 = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
    val3 = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
    val4 = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

    if st.button("Predict"):
        try:
            data = [[val1, val2, val3, val4]]
            prediction = str(model.predict(data)[0])
            pred_class = class_dict.get(prediction, "Clase desconocida")
            st.success(f"Predicción: {pred_class}")
        except Exception as e:
            st.error(f"Error al predecir: {str(e)}")

if __name__ == "__main__":
    main()