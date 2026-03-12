import os
import time
import streamlit as st
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="Poet Quote Prediction",
    page_icon="📝",
    layout="centered"
)

# ------------------------------
# Custom UI Styling
# ------------------------------

st.markdown("""
<style>

body {
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.big-title{
text-align:center;
font-size:70px;
font-weight:bold;
color:white;
letter-spacing:2px;
animation: glow 2s ease-in-out infinite alternate;
}

.subtitle{
text-align:center;
font-size:22px;
color:#d1d1d1;
margin-bottom:30px;
}

@keyframes glow{
from {text-shadow:0 0 10px #00ffd5;}
to {text-shadow:0 0 30px #00ffd5;}
}

.result-box{
background:#1e293b;
padding:25px;
border-radius:15px;
text-align:center;
font-size:26px;
color:#00ffd5;
margin-top:20px;
}

.stButton>button{
background: linear-gradient(90deg,#ff6a00,#ee0979);
color:white;
font-size:18px;
border-radius:12px;
padding:10px 25px;
border:none;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------
# Title
# ------------------------------

st.markdown('<div class="big-title">Poet Quote Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Poetry Generator</div>', unsafe_allow_html=True)

# ------------------------------
# Load Model + Tokenizer
# ------------------------------

@st.cache_resource
def load_resources():

    model_path = "lstm_model.h5"
    tokenizer_path = "tokenizer.pkl"
    maxlen_path = "max_len.pkl"

    if not os.path.exists(model_path):
        return None, None, None

    model = load_model(model_path)

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    with open(maxlen_path, "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()

if model is None:
    st.error("Model files not found. Please place model files in the app folder.")
    st.stop()

# ------------------------------
# Predict Next Word
# ------------------------------

def predictor(model, tokenizer, text, max_len):

    seq = tokenizer.texts_to_sequences([text])[0]

    if len(seq) == 0:
        return ""

    seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')

    preds = model.predict(seq, verbose=0)

    predicted_index = np.random.choice(len(preds[0]), p=preds[0])

    next_word = tokenizer.index_word.get(predicted_index, "")

    return next_word


# ------------------------------
# Generate Sentence
# ------------------------------

def generate_text(model, tokenizer, seed_text, max_len, n_words):

    for _ in range(n_words):

        next_word = predictor(model, tokenizer, seed_text, max_len)

        if next_word == "":
            break

        seed_text += " " + next_word

    return seed_text


# ------------------------------
# Generate Poem (4 lines)
# ------------------------------

def generate_poem(model, tokenizer, seed_text, max_len):

    poem = []
    current_text = seed_text

    for _ in range(4):

        line = generate_text(model, tokenizer, current_text, max_len, 8)

        poem.append(line)

        current_text = line

    return poem


# ------------------------------
# Typing Animation
# ------------------------------

def typing_effect(text):

    placeholder = st.empty()

    output = ""

    for word in text.split():

        output += word + " "

        placeholder.markdown(
            f"<div class='result-box'>{output}</div>",
            unsafe_allow_html=True
        )

        time.sleep(0.15)


# ------------------------------
# User Input
# ------------------------------

user_input = st.text_input(
    "Start a poetic sentence",
    placeholder="Example: The moon shines"
)

num_words = st.slider("Words to generate", 5, 20, 10)

# ------------------------------
# Generate Quote Button
# ------------------------------

if st.button("Generate Quote"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:

        generated_text = generate_text(model, tokenizer, user_input, max_len, num_words)

        typing_effect(generated_text)


# ------------------------------
# Generate Poem Button
# ------------------------------

if st.button("Generate Poem"):

    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:

        poem = generate_poem(model, tokenizer, user_input, max_len)

        st.markdown("### Generated Poem")

        for line in poem:

            typing_effect(line)

            time.sleep(0.5)

# ------------------------------
# Footer
# ------------------------------

st.markdown("---")
st.caption("Deep Learning • LSTM • Poetry Generator")