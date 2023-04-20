import csv
import pickle
from fastapi import FastAPI, Form, Body, Depends
from pydantic import BaseModel
from typing import Optional
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from config.db_handle import registerUser, loginUser, getUserDataById, getUserDataByEmail, get_all_resources, \
    make_ratings, export_csv, getResDataByIds
# from model.user import users
from auth.auth_bearer import JWTBearer
from auth.auth_handler import signJWT
from model.res_recommend import recommend_items
from model.train_model import recommend_items_at
from schemas.resource import Ratings
from schemas.user import UserSchema, UserLoginSchema
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

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
async def predict_learning_style(sentence: Sentence):
    # Preprocess the input sentence
    processed_sentence = preprocess_text(sentence.sentence)

    # Make a prediction with the loaded model
    prediction = model.predict(processed_sentence)

    # Decode the prediction labels using the label encoder
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=-1))[0]

    # Return the predicted label
    return {'learning_preference': predicted_label}


async def predictLearning_style(sentence):
    # Preprocess the input sentence
    processed_sentence = preprocess_text(sentence)

    # Make a prediction with the loaded model
    prediction = model.predict(processed_sentence)

    # Decode the prediction labels using the label encoder
    predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=-1))[0]

    # Return the predicted label
    return predicted_label


@app.post("/user/signup", tags=["user"])
async def create_user(user: UserSchema = Body(...)):
    learningstyle = predictLearning_style(user.sentence)
    if registerUser(user, learningstyle):
        return getUserDataById(user.email)
    else:
        return {
            "error": "Unable to register!"
        }


@app.post("/user/login", tags=["user"])
async def user_login(user: UserLoginSchema = Body(...)):
    if loginUser(user):
        return getUserDataByEmail(user.email)
    else:
        return {
            "error": "Wrong login details!"
        }


@app.get("/user/data/{id}", dependencies=[Depends(JWTBearer())], tags=["user"])
async def user_data(id: int):
    data = getUserDataById(id)
    return data


@app.get("/res/data", dependencies=[Depends(JWTBearer())], tags=["resources"])
async def res_data():
    data = get_all_resources()
    return data


@app.post("/res/rate/{res_id}/{user_id}")
async def res_rate(res_id: int, user_id: int, rate: Ratings = Body(...)):
    data = make_ratings(rate, res_id, user_id)
    return data


@app.get("/export-csv")
async def export_csv_file():
    data = export_csv()
    return data


@app.get("/res/recommendation")
def res_recommendation(user: str, n: int = 5, k: int = 5):
    # ratings_data = pd.read_csv('ratings.csv')
    # # Create user-item matrix
    # print(ratings_data)
    # user_item_matrix = ratings_data.pivot_table(index='user_id', columns='res_id', values='rate')
    # print(user_item_matrix)
    # # Fill missing values with 0
    # user_item_matrix = user_item_matrix.fillna(0)
    # print(user_item_matrix)
    # # Calculate cosine similarity between users
    # user_similarity_matrix = cosine_similarity(user_item_matrix)
    #
    # print(user_similarity_matrix)
    #
    # data = recommend_items(3, user_similarity_matrix, user_item_matrix)
    # return data
    top_n_items = recommend_items(user, n, k)
    return top_n_items


ratings = {}
learning_styles = {}
items = set()
with open('ratings.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_id = int(row['user_id'])
        res_id = int(row['res_id'])
        rate = float(row['rate'])
        learning_style = row['learning_style']
        items.add(res_id)
        if user_id not in ratings:
            ratings[user_id] = {}
        ratings[user_id][res_id] = rate
        if user_id not in learning_styles:
            learning_styles[user_id] = learning_style

# Define a function to compute the cosine similarity between two users
def cosine_sim(user1, user2):
    common_items = set(ratings[user1].keys()) & set(ratings[user2].keys())
    if len(common_items) == 0:
        return 0.0
    numerator = sum([ratings[user1][item] * ratings[user2][item] for item in common_items])
    denominator = np.sqrt(sum([ratings[user1][item]**2 for item in ratings[user1]]) * sum([ratings[user2][item]**2 for item in ratings[user2]]))
    return numerator / denominator

# Define a function to get the top k most similar users to a given user
def get_top_similar_users(user, learning_style, k=5):
    similarities = [(cosine_sim(user, other_user), other_user) for other_user in ratings if other_user != user and learning_styles.get(other_user) == learning_style]
    similarities.sort(reverse=True)
    return [other_user for similarity, other_user in similarities[:k]]

# Define a function to predict a user's rating for a given item using k-NN collaborative filtering
def predict_rating(user, item, k=5):
    if user not in ratings:
        return 0.0
    if item not in items:
        return 0.0
    if user not in learning_styles:
        return 0.0  # or return a default rating
    similar_users = get_top_similar_users(user, learning_styles[user], k)
    numerator = sum([cosine_sim(user, other_user) * (ratings[other_user].get(item, 0.0) - np.mean([rating for rating in ratings[other_user].values() if rating > 0])) for other_user in similar_users])
    denominator = sum([cosine_sim(user, other_user) for other_user in similar_users])
    if denominator == 0.0:
        return 0.0
    else:
        return np.mean([rating for rating in ratings[user].values() if rating > 0]) + (numerator / denominator)

# Define a FastAPI endpoint to get a list of recommended resources for a given user
@app.get('/recommendations/res')
def get_recommendations(user_id: int, k: int = 5):
    recommendations = []
    for item in items:
        rating = predict_rating(user_id, item, k)
        if rating > 0.0:
            recommendations.append({'res_id': item, 'rating': rating})
    recommendations.sort(reverse=True, key=lambda x: x['rating'])

    rec = recommendations[:k]
    res_list = [{'res_id': item['res_id'], 'rating': item['rating']} for item in rec]
    res_id_list = [r['res_id'] for r in rec]

    data = getResDataByIds(res_list)
    data.sort(reverse=True, key=lambda x: x['rating'])
    return data