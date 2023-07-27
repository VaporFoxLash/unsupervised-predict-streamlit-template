"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
# ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
# model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def prediction_item(item_id, model, ratings_df):
    """Map a given favourite movie to users within the MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.
    """
    # Data preprosessing
    tests = ratings_df.copy()
    tests.drop(['timestamp'], axis=1, inplace=True)
    tests = tests.head(20000)
    
    reader = Reader(rating_scale=(0.5, 5))
    load_df = Dataset.load_from_df(tests[['userId', 'movieId', 'rating']], reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose=False))
    return predictions

def pred_movies(movie_list, model, ratings_df):
    """Maps the given favourite movies selected within the app to corresponding users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.
    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id=i, model=model, ratings_df=ratings_df)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    # Loading SVD model
    svd_model = pickle.load(open('resources/models/SVD.pkl', 'rb'))

    # Get the list of all movie titles
    all_movie_titles = movies_df['title'].unique()

    # Get the movie IDs for the movies in movie_list
    movie_ids = []
    for title in movie_list:
        if title in movies_df['title'].values:
            movie_id = movies_df[movies_df['title'] == title]['movieId'].values[0]
            movie_ids.append(movie_id)

    print(f"Movie IDs: {movie_ids}") # Debug line

    # Find the users who have the highest rating for the movies in movie_list
    user_ids = pred_movies(movie_list, model=svd_model, ratings_df=ratings_df)

    print(f"User IDs: {user_ids}") # Debug line

    # Predict the ratings for all movies
    predicted_ratings = []
    for title in all_movie_titles:
        if title in movie_list or title not in all_movie_titles:
            continue
        movie_id = movies_df[movies_df['title'] == title]['movieId'].values[0]
        for user_id in user_ids:
            if user_id in svd_model.trainset.all_users(): # ensure the user ID exists in the model's training data
                predicted_rating = svd_model.predict(user_id, movie_id).est
                predicted_ratings.append((title, predicted_rating))

    # Sort the predicted ratings in descending order
    predicted_ratings.sort(key=lambda x: x[1], reverse=True)

    # Get the top n movie titles
    recommended_movies = [title for title, rating in predicted_ratings[:top_n]]

    return list(set(recommended_movies))


