import pickle

from fastapi import FastAPI, Form
from pydantic import BaseModel
from typing import Optional
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to LearningRes_RecMaster"}


# # Load the saved model and tokenizer
# model = tf.keras.models.load_model('LearningStyleClassifier.h5')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#
# # Load the label encoder to map output classes
# with open('labelEncoder.pickle', 'rb') as f:
#     label_encoder = pickle.load(f)
#
# # with open('tokenizer.pickle', 'rb') as handle:
# #     tokenizer = pickle.load(handle)
#
# # Define the input and output schemas for the API
# class InputSchema(BaseModel):
#     sentence: str
#
#
# class OutputSchema(BaseModel):
#     prediction: str
#
#
# # Define the prediction endpoint
# @app.post('/predict', response_model=OutputSchema)
# async def predict(input_data: InputSchema):
#     # Tokenize the input sentence using the BERT tokenizer
#     encoded_sentence = tokenizer.encode_plus(
#         input_data.sentence,
#         add_special_tokens=True,
#         max_length=512,
#         padding='max_length',
#         return_attention_mask=True,
#         return_tensors='tf'
#     )
#
#     #encoded_sentence = tokenizer.texts_to_sequences(input_data.sentence)
#
#     # Make the prediction using the loaded model
#     prediction = model.predict(encoded_sentence)
#
#     # Map the prediction to the corresponding label
#     label = label_encoder.inverse_transform(np.argmax(prediction))
#
#     # Return the predicted label
#     return {'prediction': label}


# Load the saved model
with open('learningstylepredictor.pickle', 'rb') as f:
    model = pickle.load(f)


# Define the route for the prediction API
@app.post('/predict/test')
async def predict(answers: str = Form(...)):
    # Parse the answers and convert them to a numpy array
    answers_list = answers.split(',')
    answers_array = np.array(answers_list).reshape(1, -1)

    # Make the prediction using the loaded model
    prediction = model.predict(answers_array)

    # Return the predicted learning preference
    return {'learning_preference': prediction[0]}






# Load the tokenizer and label encoder from pickle files
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('labelEncoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load the trained model from the h5 file
model = tf.keras.models.load_model('LearningStyleClassifier.h5')

# Define the input data schema
class Sentence(BaseModel):
    sentence: str

# Define a function to preprocess the input text and tokenize it
def preprocess_text(text):
    text = text.lower()
    text = tokenizer.texts_to_sequences([text])
    text = tf.keras.preprocessing.sequence.pad_sequences(text, maxlen=48, padding='post')
    return text

# Define the prediction endpoint
@app.post('/predict')
def predict_learning_style(sentence: Sentence):
    # Preprocess the input sentence
    processed_sentence = preprocess_text(sentence.sentence)

    # Make a prediction with the loaded model
    prediction = model.predict(processed_sentence)

    # Decode the prediction labels using the label encoder
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=-1))

    # Return the predicted label
    return {'learning_preference': predicted_label[0]}