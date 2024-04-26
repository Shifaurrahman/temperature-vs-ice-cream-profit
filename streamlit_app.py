import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from keras.models import load_model # type: ignore


@st.cache(allow_output_mutation=True)
def load_nn_model():
    try:
        model = load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def predict_profit(model, temperature):
    try:
        temperature_fahrenheit = temperature
        prediction = model.predict(np.array([[temperature_fahrenheit]]))
        predicted_profit = prediction[0][0]
        return predicted_profit
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def main():
    st.title('Temperature vs Ice Cream Profit Predictor')
    
    model = load_nn_model()

    if model is not None:
        temperature = st.number_input('Enter Temperature (Fahrenheit):', step=0.01)
        
        if st.button("Predict Profit"):
            profit = predict_profit(model, temperature)
            if profit is not None:
                st.success(f"Predicted Profit: {profit:.2f}")


if __name__ == '__main__':
    main()
