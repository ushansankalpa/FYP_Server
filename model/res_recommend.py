import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# # Load data
# ratings_data = pd.read_csv('ratings.csv')
#
# # Create user-item matrix
# user_item_matrix = ratings_data.pivot_table(index='user_id', columns='item_id', values='rating', fill_value='learning_style')
#
# # Fill missing values with 0
# user_item_matrix = user_item_matrix.fillna(0)
#
# # Calculate cosine similarity between users
# user_similarity_matrix = cosine_similarity(user_item_matrix)


# Create a function to recommend items to a user
def recommend_items(user_id, user_similarity_matrix, user_item_matrix):
    # Calculate weighted average of ratings for similar users
    print( user_id)
    print(66 + user_similarity_matrix)
    print(77 + user_item_matrix)
    similar_users = user_similarity_matrix[user_id - 1]
    print(similar_users)
    user_ratings = user_item_matrix.loc[user_id]
    print(user_ratings)
    similar_user_ratings = user_item_matrix.mul(similar_users, axis=0).sum(axis=1) / similar_users.sum()

    print(similar_user_ratings)
    # Exclude items already rated by the user
    recommended_items = similar_user_ratings.drop(user_ratings.index)
    print(recommended_items)

    # Sort recommended items by rating
    recommended_items = recommended_items.sort_values(ascending=False)
    print(recommended_items)

    return recommended_items.head(10)

# Test the function
