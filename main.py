import pickle
from fastapi import FastAPI, Form, Body, Depends
from pydantic import BaseModel
from typing import Optional
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from config.db_handle import registerUser, loginUser, getUserDataById
# from model.user import users
from auth.auth_bearer import JWTBearer
from auth.auth_handler import signJWT
from schemas.user import UserSchema, UserLoginSchema
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to LearningRes_RecMaster"}



# Load the saved model
with open('learningstylepredictor.pickle', 'rb') as f:
    model_test = pickle.load(f)


@app.post('/predict/test')
async def predict(answers: list[int]):
    # Make the prediction using the loaded model
    prediction = model_test.predict([answers])

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
    text = tf.keras.preprocessing.sequence.pad_sequences(text, maxlen=48, truncating='pre')
    return text


# Define the prediction endpoint
@app.post('/predict')
def predict_learning_style(sentence: Sentence):
    # Preprocess the input sentence
    processed_sentence = preprocess_text(sentence.sentence)

    # Make a prediction with the loaded model
    prediction = model.predict(processed_sentence)

    # Decode the prediction labels using the label encoder
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=-1))[0]

    # Return the predicted label
    return {'learning_preference': predicted_label}


def predictLearning_style(sentence):
    # Preprocess the input sentence
    processed_sentence = preprocess_text(sentence)

    # Make a prediction with the loaded model
    prediction = model.predict(processed_sentence)

    # Decode the prediction labels using the label encoder
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=-1))[0]

    # Return the predicted label
    return predicted_label


@app.post("/user/signup", tags=["user"])
def create_user(user: UserSchema = Body(...)):
    learningstyle = predictLearning_style(user.sentence)
    if registerUser(user, learningstyle):
        return getUserDataById(user.email)
    else:
        return {
            "error": "Unable to register!"
        }

@app.post("/user/login", tags=["user"])
def user_login(user: UserLoginSchema = Body(...)):
    if loginUser(user):
        return getUserDataById(user.email)
    else:
        return {
            "error": "Wrong login details!"
        }


@app.get("/user/data", tags=["user"])
def user_data(user_id):
    data = getUserDataById(user_id)
    return data
