# Libraries for data preparation & visualization
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio

pio.renderers.default = "png"

# Ignore printing warnings for general readability
import warnings

warnings.filterwarnings("ignore")

# pip install scikit-surprise
# Importing libraries for model building & evaluation
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, SVD
from surprise import accuracy

# dataset = pd.read_csv('ratings.csv')
# def loaddata(filename):
#     df = pd.read_csv(f'{filename}.csv',sep=';',error_bad_lines=False,warn_bad_lines=False,encoding='latin-1')
#     return df
#
# # book   = loaddata("BX-Books")
# # user   = loaddata("BX-Users")
# rating = loaddata("ratings")
#
#
# rating = rating[rating['User-ID'].isin(rating_users[rating_users['Rating']>250]['index'])]
# rating = rating[rating['ISBN'].isin(rating_books[rating_books['Rating']> 50]['index'])]
#
# rating

from fastapi import FastAPI
from typing import List
from collections import defaultdict
import pandas as pd
import numpy as np


# app = FastAPI()

# Load the CSV file into a pandas DataFrame
# df = pd.read_csv('ratings.csv')
#
# # Convert the DataFrame to a user-item rating matrix
# ratings = defaultdict(dict)
# for i, row in df.iterrows():
#     ratings[row['user_id']][row['res_id']] = row['rate']

# Define a function to compute the cosine similarity between two users
def cosine_sim(user1, user2, ratings):
    common_items = set(ratings[user1].keys()) & set(ratings[user2].keys())
    if len(common_items) == 0:
        return 0.0
    numerator = sum([ratings[user1][item] * ratings[user2][item] for item in common_items])
    denominator = np.sqrt(sum([ratings[user1][item] ** 2 for item in ratings[user1]]) * sum(
        [ratings[user2][item] ** 2 for item in ratings[user2]]))
    return numerator / denominator


# Define a function to get the top-N similar users to a given user
def get_top_similar_users(user, ratings, learning_styles , n=5):
    similarities = [(cosine_sim(ratings[user], ratings[other_user]), other_user) for other_user in ratings if
                    other_user != user and learning_styles[other_user] == learning_style]
    similarities.sort(reverse=True)
    return similarities[:n]


# Define a function to recommend items for a given user
def recommend_items_at(user, n=5, k=5):
    df = pd.read_csv('ratings.csv')

    # Convert the DataFrame to a user-item rating matrix
    ratings = defaultdict(dict)
    learning_styles = {}
    for i, row in df.iterrows():
        user = row['user_id']
        item = row['res_id']
        rate = row['rate']
        learning_style = row['learning_style']
        ratings[user][item] = rate
        learning_styles[user] = learning_style

    print(ratings)

    user_ratings = ratings[user]
    predicted_ratings = {}
    for item in ratings:
        if item not in user_ratings:
            predicted_ratings[item] = predict_rating(user, item, ratings, learning_styles[user], k)
    sorted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
    top_n_items = [x[0] for x in sorted_ratings[:n]]
    return top_n_items


# Define a function to predict the rating of a user for an item based on the ratings of similar users
def predict_rating(user, item, ratings, learning_styles, k=5):
    similar_users = get_top_similar_users(user, ratings ,learning_styles, k)
    numerator = sum([similarity * ratings[other_user][item] for similarity, other_user in similar_users])
    denominator = sum([similarity for similarity, other_user in similar_users])
    if denominator == 0:
        return 0.0
    return numerator / denominator
