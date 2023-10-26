from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np


# create a flask app
app = Flask(__name__)

# load the trained model
model = load_model('NLP_airline_model.h5')

# load the saved tokenizer
with open ('tokenizer.pkl', 'rb') as tk:
    tokenizer = pickle.load(tk)

# Define Preprocessing Function for user's text input
def preprocess_text(text):
    # tokenize text
    tokens = tokenizer.texts_to_sequences([text])

    # pad the sequnces to a fixed length
    padded_tokens = pad_sequences(tokens, maxlen = 100)
    return padded_tokens[0]


@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    pred_sentiment = ''

    if request.method == 'POST':
        user_input = request.form['user_input']

        # preprocess user input
        preprocessed_input = preprocess_text(user_input)

        # make prediction using the model 
        prediction = model.predict(np.array([preprocessed_input]))

        # determine the class with the highest probability
        predicted_class = np.argmax(prediction)
        
        # map the predicted class index to the corresponding label
        class_labels = ['Negative', 'Positive', 'Neutral']
        pred_sentiment = class_labels[predicted_class]

    return render_template('index.html', sentiment = pred_sentiment)


if __name__ == '__main__':
    app.run(debug = True)