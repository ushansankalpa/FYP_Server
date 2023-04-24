import csv
import pickle
from fastapi import FastAPI, Form, Body, Depends
from pydantic import BaseModel
from typing import Optional
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from starlette.responses import JSONResponse
from transformers import BertTokenizer
from config.db_handle import registerUser, loginUser, getUserDataById, getUserDataByEmail, get_all_resources, \
    make_ratings, export_csv, getResDataByIds, updateUserLearninStyle, get_all_users, get_learning_styles, \
    get_similar_users, insert_resources, getUserDataByEmail_Auth
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
import mysql.connector

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


@app.post('/questionary/result/{user_id}')
def predict_by_test(user_id:int, answers: list[int]):
    # Make the prediction using the loaded model
    prediction = model_test.predict([answers])
    print(round(prediction[0]))
    learning_style = ''
    if round(prediction[0]) == 0:
        learning_style = 'Auditory'
    elif round(prediction[0]) == 1:
        learning_style = 'Kinesthetic'
    elif round(prediction[0]) == 2:
        learning_style = 'Visual'

    data = updateUserLearninStyle(learning_style, user_id)

    # Return the predicted learning preference
    return data


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
        return getUserDataByEmail_Auth(user.email)
    else:
        return {
            "error": "Unable to register!"
        }


# @app.post("/questionary/result/{user_id}")
# async def questionary_result(data: list[int]):
#     prediction = model_test.predict([data])
#     return prediction
#     # if update(user_id, learningstyle):
#     #     return getUserDataById(user.email)
#     # else:
#     #     return {
#     #         "error": "Unable to register!"
#     #     }


@app.post("/user/login", tags=["user"])
def user_login(user: UserLoginSchema = Body(...)):
    if loginUser(user):
        return getUserDataByEmail(user.email)
    else:
        return {
            "error": "Wrong login details!"
        }


@app.get("/user/data/{id}", dependencies=[Depends(JWTBearer())], tags=["user"])
def user_data(id: int):
    data = getUserDataById(id)
    return data


@app.get("/res/data", dependencies=[Depends(JWTBearer())], tags=["resources"])
def res_data():
    data = get_all_resources()
    return data

@app.post("/post/data", tags=["resources"])
def res_data(res = Body(...)):
    data = insert_resources(res)
    if data:
        return data
    else:
        return {
            "error": "Unable to Insert data!"
        }
@app.get("/user/data", tags=["user"])
def user_data():
    data = get_all_users()
    return data

@app.get("/get/learning_styles")
def user_data():
    data = get_learning_styles()
    return data

@app.post("/res/rate/{res_id}/{user_id}")
def res_rate(res_id: int, user_id: int, rate: Ratings = Body(...)):
    data = make_ratings(rate, res_id, user_id)
    return data


@app.get("/export-csv")
def export_csv_file():
    data = export_csv()
    return data





ratings = {}
learning_styles = {}
items = set()
def read_csv():
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
    print(user1)
    common_items = set(ratings[user1].keys()) & set(ratings[user2].keys())
    if len(common_items) == 0:
        return 0.0
    numerator = sum([ratings[user1][item] * ratings[user2][item] for item in common_items])
    denominator = np.sqrt(sum([ratings[user1][item]**2 for item in ratings[user1]]) * sum([ratings[user2][item]**2 for item in ratings[user2]]))
    return numerator / denominator

# Define a function to get the top k most similar users to a given user
def get_top_similar_users(user, learning_style, k=1000):
    similarities = [(cosine_sim(user, other_user), other_user) for other_user in ratings if other_user != user and learning_styles.get(other_user) == learning_style]
    similarities.sort(reverse=True)
    return [other_user for similarity, other_user in similarities[:k]]

# Define a function to predict a user's rating for a given item using k-NN collaborative filtering
def predict_rating(user, item, k=1000):
    if user not in ratings:
        return 0.0
    if item not in items:
        return 0.0
    if user not in learning_styles:
        return 0.0  # or return a default rating
    similar_users = get_top_similar_users(user, learning_styles[user], k)
    print(learning_styles[user])
    numerator = sum([cosine_sim(user, other_user) * (ratings[other_user].get(item, 0.0) - np.mean([rating for rating in ratings[other_user].values() if rating > 0])) for other_user in similar_users])
    denominator = sum([cosine_sim(user, other_user) for other_user in similar_users])
    if denominator == 0.0:
        return 0.0
    else:
        return np.mean([rating for rating in ratings[user].values() if rating > 0]) + (numerator / denominator)


# Define a FastAPI endpoint to get a list of recommended resources for a given user
@app.get('/recommendations/res')
def get_recommendations(user_id: int, k: int = 1000):
    read_csv()
    if user_id not in ratings:
        recommendations = []
        for item in items:
            rating = predict_rating(user_id, item, k)
            if rating > 0.0:
                recommendations.append({'res_id': item, 'rating': rating})
        recommendations.sort(reverse=True, key=lambda x: x['rating'])

        if len(recommendations) == 0:
            return {'message': 'No recommendations available for this user'}

        rec = recommendations[:k]
        res_list = [{'res_id': item['res_id'], 'rating': item['rating']} for item in rec]
        data = getResDataByIds(res_list)
        return data
    else:
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



# @app.get("/res/recommendation")
# def res_recommendation(user_id: int, n: int = 5, k: int = 5):
#     recommendations = []
#     for item in items:
#         rating = predict_rating(user_id, item, k)
#         if rating > 0.0:
#             recommendations.append({'res_id': item, 'rating': rating})
#     recommendations.sort(reverse=True, key=lambda x: x['rating'])
#
#     rec = recommendations[:k]
#     print(rec)
#     res_list = [{'res_id': item['res_id'], 'rating': item['rating']} for item in rec]
#     res_id_list = [r['res_id'] for r in rec]
#
#     data = getResDataByIds(res_list)
#     print(data)
#     # if data:
#     #     data.sort(reverse=True, key=lambda x: x['rating'])
#     return data
#
#     # ratings_data = pd.read_csv('ratings.csv')
#     # # Create user-item matrix
#     # print(ratings_data)
#     # user_item_matrix = ratings_data.pivot_table(index='user_id', columns='res_id', values='rate')
#     # print(user_item_matrix)
#     # # Fill missing values with 0
#     # user_item_matrix = user_item_matrix.fillna(0)
#     # print(user_item_matrix)
#     # # Calculate cosine similarity between users
#     # user_similarity_matrix = cosine_similarity(user_item_matrix)
#     #
#     # print(user_similarity_matrix)
#     #
#     # data = recommend_items(3, user_similarity_matrix, user_item_matrix)
#     # return data
#     # top_n_items = recommend_items(user, n, k)
#     # return top_n_items
def read_ratings_data():
    ratings_df = pd.read_csv('ratings.csv')
    return ratings_df

# Define a function to get top k similar users based on learning style
def get_top_similar_users(user_id, learning_style, k):

    similar_users = get_similar_users(user_id, learning_style, k)
    return similar_users

# Define a function to calculate cosine similarity between users
def cosine_sim(user1, user2, ratings_df):
    user1_vec = ratings_df.loc[ratings_df['user_id'] == user1, 'rate'].values.reshape(1, -1)
    user2_vec = ratings_df.loc[ratings_df['user_id'] == user2, 'rate'].values.reshape(1, -1)
    if user1_vec.shape[1] > 0 and user2_vec.shape[1] > 0:
        if user1_vec.shape[1] > user2_vec.shape[1]:
            user1_vec = user1_vec[:, :user2_vec.shape[1]]
        elif user2_vec.shape[1] > user1_vec.shape[1]:
            user2_vec = user2_vec[:, :user1_vec.shape[1]]
        sim = cosine_similarity(user1_vec, user2_vec)[0][0]
        print(sim)
        return sim
    else:
        return 0.0
    #sim = cosine_similarity(user1_vec, user2_vec)[0][0]
    # if user1_vec.size > 0 and user2_vec.size > 0:
    #     if user1_vec.shape != user2_vec.shape:
    #         user2_vec = user2_vec.reshape(user1_vec.shape)
    #     sim = cosine_similarity(user1_vec.reshape(1, -1), user2_vec.reshape(1, -1))[0][0]
    # else:
    #     sim = 0.0

# def cosine_sim_2(user1, user2, ratings_df):
#     user1_vec = get_user_vector(user1, ratings_df)
#     user2_vec = get_user_vector(user2, ratings_df)
#     if len(user2_vec) == 0:
#         return 0.0
#     else:
#         user2_vec = user2_vec.reshape(user1_vec.shape)
#         return np.dot(user1_vec, user2_vec) / (np.linalg.norm(user1_vec) * np.linalg.norm(user2_vec))

# Define a function to predict the rating for an item
def predict_rating(user_id, item_id, k=1000):
    ratings_df = read_ratings_data()
    similar_users = get_top_similar_users(user_id, ratings_df.loc[ratings_df['user_id'] == user_id, 'learning_style'].values[0], k)
    print(similar_users)
    #numerator = sum(cosine_sim(user_id, other_user, ratings_df) * (ratings_df.loc[(ratings_df['user_id'] == other_user) & (ratings_df['res_id'] == item_id), 'rate'].values[0] - np.mean(ratings_df.loc[ratings_df['user_id'] == other_user, 'rate'].values)) for other_user in similar_users)
    #denominator = sum(cosine_sim(user_id, other_user, ratings_df) for other_user in similar_users)
    numerator = sum(cosine_sim(user_id, other_user, ratings_df) * (ratings_df.loc[(ratings_df['user_id'] == other_user) & (ratings_df['res_id'] == item_id), 'rate'].values[0] - np.mean(ratings_df.loc[ratings_df['user_id'] == other_user, 'rate'].values)) if ratings_df.loc[(ratings_df['user_id'] == other_user) & (ratings_df['res_id'] == item_id), 'rate'].shape[0] > 0 else 0 for other_user in similar_users)
    denominator = sum(cosine_sim(user_id, other_user, ratings_df) for other_user in similar_users)
    if denominator == 0.0:
        return 0.0
    else:
        predicted_rating = np.mean(ratings_df.loc[ratings_df['res_id'] == item_id, 'rate']) + (numerator / denominator)
        return predicted_rating

# Define API endpoint for recommending items based on user ID
# @app.get("/recommend/{user_id}")
# async def recommend_items(user_id: int):
#     ratings_df = read_ratings_data()
#     items_rated = set(ratings_df.loc[ratings_df['user_id'] == user_id, 'res_id'])
#     items_not_rated = set(ratings_df['res_id']) - items_rated
#     predicted_ratings = {}
#     for item_id in items_not_rated:
#         predicted_rating = predict_rating(user_id, item_id)
#         predicted_ratings[item_id] = predicted_rating
#     sorted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
#     recommended_items = [{'res_id': item_id, 'predicted_rating': predicted_rating} for item_id, predicted_rating in sorted_ratings]
#     return recommended_items



@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int):
    # Load ratings data from CSV
    ratings_df = pd.read_csv("ratings.csv")

    # Load user data from MySQL
    cnx = mysql.connector.connect(host="localhost",
        user="root",
        password="",
        database="learnMaster")
    user_df = pd.read_sql_query("SELECT * FROM user WHERE id != %s", cnx, params=[user_id])

    print(user_df)
    le = LabelEncoder()
    user_df['learning_style_encoded'] = le.fit_transform(user_df['learning_style'])

    # Define cosine similarity function with encoded values
    def cosine_sim(x, y):
        x = x.values.reshape(1, -1)
        y = y.values.reshape(1, -1)
        return cosine_similarity(x, y)[0][0]

    # Calculate cosine similarity between users based on their learning style
    user_df['similarity'] = user_df.apply(
        lambda row: cosine_sim_2(row[['learning_style_encoded']], user_df[['learning_style_encoded']]), axis=1)

    # Get the top 10 most similar users
    similar_user_ids = user_df.sort_values('similarity', ascending=False)['id'].values[1:11]

    print(similar_user_ids)

    # Filter ratings data to only include items rated by similar users
    similar_ratings_df = ratings_df[ratings_df['user_id'].isin(similar_user_ids)]

    # Group by item ID and compute the mean rating for each item
    item_ratings = similar_ratings_df.groupby('res_id')['rate'].mean()

    # Load resource data from MySQL
    resource_df = pd.read_sql_query("SELECT * FROM resource", cnx)

    # Merge item ratings and resource data to get recommended resources
    recommendations_df = pd.merge(item_ratings, resource_df, on='res_id')
    print(recommendations_df)

    # If the target user has no ratings in the ratings CSV, predict their ratings using the mean rating for each item
    if user_id not in ratings_df['user_id'].unique():
        predicted_ratings = pd.DataFrame(columns=['res_id', 'rate'])
        for item_id in resource_df['res_id']:
            mean_rating = item_ratings.get(item_id, resource_df['rate'].mean())
            predicted_ratings = predicted_ratings.append({'res_id': item_id, 'rate': mean_rating}, ignore_index=True)
            print(predicted_ratings)
        recommendations_df = pd.merge(predicted_ratings, resource_df, on='res_id')
        print(recommendations_df)

    # Sort recommendations by predicted rating (or mean rating, if predicting) and return the top 10
    sorted_recommendations = recommendations_df.sort_values('rate', ascending=False)
    top_recommendations = sorted_recommendations.head(10)
    res_list = [{'res_id': res_id, 'rating': rating} for res_id, rating in
                zip(top_recommendations['res_id'], top_recommendations['rate'])]
    res_ids_and_rates = top_recommendations.loc[:, ['res_id', 'rate']]
    print(top_recommendations)
    return res_list


def cosine_sim_2(user1, user2):
    user1_vec = user1.values.reshape(1, -1)
    user2_vec = user2.values.reshape(1, -1)
    if user1_vec.shape[1] > 0 and user2_vec.shape[1] > 0:
        if user1_vec.shape[1] > user2_vec.shape[1]:
            user1_vec = user1_vec[:, :user2_vec.shape[1]]
        elif user2_vec.shape[1] > user1_vec.shape[1]:
            user2_vec = user2_vec[:, :user1_vec.shape[1]]
        sim = cosine_similarity(user1_vec, user2_vec)[0][0]
        return sim
    else:
        return 0.0





















# define a function that will find similar users based on their learning style
def find_similar_users(user_id):
  cnx = mysql.connector.connect(host="localhost",
                                  user="root",
                                  password="",
                                  database="learnMaster")
  # get the learning style of the current user
  cursor = cnx.cursor()
  query = "SELECT learning_style FROM user WHERE id = %s"
  cursor.execute(query, (user_id,))
  learning_style = cursor.fetchone()[0]

  # find all users with the same learning style
  query = "SELECT id FROM user WHERE learning_style = %s AND id != %s"
  cursor.execute(query, (learning_style, user_id))
  similar_users = [x[0] for x in cursor.fetchall()]

  return similar_users

# define a route that will accept a user_id and return a list of recommended resources
@app.get('/res/recommendations/{user_id}')
def get_recommendations_2(user_id: int):
  cnx = mysql.connector.connect(host="localhost",
                                  user="root",
                                  password="",
                                  database="learnMaster")
  # load the ratings data from the CSV file
  ratings_df = pd.read_csv('ratings.csv')

  # filter the ratings data to only include users that are similar to the current user
  similar_users = find_similar_users(user_id)
  similar_ratings_df = ratings_df[ratings_df['user_id'].isin(similar_users)]

  # calculate the cosine similarity between the current user and each similar user
  similarity_matrix = cosine_similarity(
    similar_ratings_df.pivot_table(index='user_id', columns='res_id', values='rate', fill_value=0)
  )
  user_similarity = similarity_matrix[-1][:-1]

  # calculate the weighted average of the ratings for each resource
  # ratings_matrix = similar_ratings_df.pivot_table(index='user_id', columns='res_id', values='rate', fill_value=0)
  # print(ratings_matrix)
  # weighted_ratings = (ratings_matrix.T * user_similarity).T
  # weighted_average_ratings = weighted_ratings.sum(axis=0) / user_similarity.sum()

  ratings_matrix = similar_ratings_df.pivot_table(index='user_id', columns='res_id', values='rate', fill_value=0)
  user_similarity_series = pd.Series(user_similarity, index=ratings_matrix.index[:len(user_similarity)])
  weighted_ratings = ratings_matrix.mul(user_similarity_series, axis=0)
  weighted_average_ratings = weighted_ratings.sum(axis=0) / user_similarity_series.sum()

  # convert the weighted average ratings to a pandas Series and drop any NaN values
  weighted_average_ratings = weighted_average_ratings.to_frame('rating').reset_index()
  weighted_average_ratings = weighted_average_ratings.dropna()






  # load the resource data from the MySQL database
  cursor = cnx.cursor()
  query = "SELECT res_id, title FROM resource"
  cursor.execute(query)
  resources = {res_id: res_id for (res_id, title) in cursor.fetchall()}

  # get the recommended resources
  recommendations = []
  for res_id, rating in weighted_average_ratings[['res_id', 'rating']].itertuples(index=False):
    if pd.isna(rating):
      continue
    recommendations.append({
      'resource_name': resources[res_id],
      'rating': rating
    })

  # sort the recommended resources by rating (descending order)
  recommendations = sorted(recommendations, key=lambda x: x['rating'], reverse=True)

  return recommendations





ratings_df = pd.read_csv('ratings.csv')

cnx = mysql.connector.connect(host="localhost",
        user="root",
        password="",
        database="learnMaster")
user_df = pd.read_sql_query('SELECT * From user', cnx)

# Load the user table from MySQL into a pandas DataFrame
# engine = create_engine('mysql+pymysql://username:password@localhost/db_name')
# user_df = pd.read_sql_table('user', engine)

# Define a function to get similar users based on their learning_style
def get_similar_users(user_id):
    # Get the learning_style of the user
    user_style = user_df.loc[user_df['id'] == user_id, 'learning_style'].values[0]
    # Get the users with the same learning_style
    similar_users = user_df.loc[user_df['learning_style'] == user_style, 'id']
    # Remove the current user from the list of similar users
    similar_users = similar_users[similar_users != user_id]
    return similar_users

# Define a function to recommend items based on similar users' ratings
def recommend_items(user_id):
    # Check if the user_id exists in ratings.csv
    if user_id in ratings_df['user_id'].unique():
        # Get the similar users based on their learning_style
        similar_users = get_similar_users(user_id)
        # Get the items rated by the similar users
        similar_items = ratings_df[ratings_df['user_id'].isin(similar_users)]
        # Get the items that the current user has not rated
        unrated_items = similar_items[~similar_items['res_id'].isin(ratings_df[ratings_df['user_id'] == user_id]['res_id'])]
        # Get the average rating for each item
        item_ratings = unrated_items.groupby('res_id')['rate'].mean()
        # Sort the items based on their average rating
        recommended_items = item_ratings.sort_values(ascending=False)
        return recommended_items
    else:
        encoder = OneHotEncoder()

        # Encode the learning_style column
        learning_style_encoded = encoder.fit_transform(user_df[['learning_style']]).toarray()

        # Cluster the users based on their learning_style using k-means algorithm
        kmeans = KMeans(n_clusters=3)
        user_df['cluster'] = kmeans.fit_predict(learning_style_encoded)
        # Get the users in the same cluster as the current user
        cluster_users = user_df[user_df['user_id'] != user_id][user_df['cluster'] == user_df.loc[user_df['user_id'] == user_id, 'cluster'].values[0]]
        # Get the resources recommended by the users in the same cluster
        cnx = mysql.connector.connect(host="localhost",
                                      user="root",
                                      password="",
                                      database="learnMaster")
        resource_df = pd.read_sql_query(
            "SELECT * FROM resource WHERE user_id IN ({})".format(','.join(str(x) for x in cluster_users['user_id'])), cnx)
        # Get the average rating for each resource
        resource_ratings = resource_df.groupby('res_id')['rate'].mean()
        # Sort the resources based on their average rating
        recommended_items = resource_ratings.sort_values(ascending=False)
        return recommended_items

@app.get("/recommend_items/{user_id}")
async def read_root(user_id: int):
    recommended_items = recommend_items(user_id)
    # res_list = [{'res_id': res_id, 'rating': rating} for res_id, rating in
    #             zip(recommended_items['res_id'], recommended_items['rate'])]
    print(recommended_items)
    return {"recommended_items"}