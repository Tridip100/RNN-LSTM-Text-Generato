# Next Word  Prediction

A Deep Learning project that generates poetic text using **RNN with LSTM**.
The model predicts the **next words in a sentence** and can generate short quotes or poetic lines from a given input.

Model

* Architecture: **Recurrent Neural Network (RNN) with LSTM**
* Framework: **TensorFlow / Keras**
* Training: **200 epochs**
* Task: **Next word / sentence prediction**

## Training

The model was trained on **Google Colab** because the local system was not compatible with TensorFlow training.
Recent Python versions also have compatibility issues with some TensorFlow builds, so Colab was used for a stable environment.

## Web App

A **Streamlit application** is used to interact with the model.
Users can enter a sentence and the model generates the next words or a poetic quote.

## Run the App

Install dependencies:

```
pip install streamlit tensorflow numpy
```

Run the application:

```
streamlit run app.py
```

## Files

* `app.py` – Streamlit web application
* `lstm_model.h5` – trained model
* `tokenizer.pkl` – tokenizer used during training
* `max_len.pkl` – sequence length
