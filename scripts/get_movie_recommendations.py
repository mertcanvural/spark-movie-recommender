#!/usr/bin/env python3
"""
Interactive script to get movie recommendations by entering movie IDs.
Loads the trained Spark ALS model and provides recommendations for user-entered movie IDs.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# Initialize Spark session
spark = (
    SparkSession.builder.appName("Movie Recommendations")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

# Paths
MODEL_PATH = "models/spark_als_model"
MOVIES_PATH = "data/processed/movies.csv"


def load_model():
    """Load the trained ALS model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print(
            "Please train the model first using 'python scripts/train_spark_model.py'"
        )
        sys.exit(1)

    try:
        model = ALSModel.load(MODEL_PATH)
        print(f"Successfully loaded model from {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)


def load_movies():
    """Load the movies data"""
    if not os.path.exists(MOVIES_PATH):
        print(f"Error: Movies data not found at {MOVIES_PATH}")
        sys.exit(1)

    try:
        # Load movies data
        movies_df = pd.read_csv(MOVIES_PATH)
        print(f"Loaded {len(movies_df)} movies from {MOVIES_PATH}")
        return movies_df
    except Exception as e:
        print(f"Error loading movies data: {str(e)}")
        sys.exit(1)


def get_movie_details(movie_id, movies_df):
    """Get movie details by movie ID"""
    movie = movies_df[movies_df["movieId"] == movie_id]
    if len(movie) == 0:
        return None
    return movie.iloc[0]


def get_recommendations(model, movie_id, num_recommendations=10):
    """Get movie recommendations based on item similarity"""
    # Create a dataframe with a single item
    item_df = spark.createDataFrame([(movie_id,)], ["movieId"])

    # Get recommendations
    try:
        recommendations = model.recommendForItemSubset(item_df, num_recommendations)
        return recommendations
    except Exception as e:
        print(f"Error getting recommendations: {str(e)}")
        return None


def format_recommendations(recommendations, movies_df):
    """Format recommendations for display"""
    if recommendations is None or recommendations.count() == 0:
        return "No recommendations found."

    # Extract recommendations
    rec_list = recommendations.collect()[0]["recommendations"]

    # Format results
    results = []
    for rec in rec_list:
        movie_id = rec["movieId"]
        score = rec["rating"]
        movie = get_movie_details(movie_id, movies_df)

        if movie is not None:
            title = movie["title"]
            genres = movie["genres"]
            results.append(
                f"Movie ID: {movie_id}, Title: {title}, Score: {score:.2f}, Genres: {genres}"
            )

    return "\n".join(results)


def interactive_recommendations():
    """Interactive loop for getting recommendations"""
    print("Loading model and movie data...")
    model = load_model()
    movies_df = load_movies()

    print("\n" + "=" * 80)
    print("Movie Recommendation System")
    print("=" * 80)
    print("Enter a movie ID to get recommendations, or 'q' to quit.")
    print("You can see available movies by entering 'list'.")

    while True:
        try:
            user_input = input(
                "\nEnter movie ID (or 'q' to quit, 'list' to see movies): "
            )

            if user_input.lower() == "q":
                break

            if user_input.lower() == "list":
                # Show a sample of movies
                print("\nSample of available movies:")
                sample = movies_df.sample(10)
                for _, row in sample.iterrows():
                    print(f"ID: {row['movieId']}, Title: {row['title']}")
                continue

            try:
                movie_id = int(user_input)
            except ValueError:
                print("Please enter a valid movie ID (integer).")
                continue

            # Check if movie exists
            movie = get_movie_details(movie_id, movies_df)
            if movie is None:
                print(f"Movie with ID {movie_id} not found.")
                continue

            print(f"\nGetting recommendations for: {movie['title']} (ID: {movie_id})")
            recommendations = get_recommendations(model, movie_id)

            print("\nRecommended Movies:")
            print("-" * 80)
            print(format_recommendations(recommendations, movies_df))

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

    print("\nThank you for using the Movie Recommendation System!")
    spark.stop()


if __name__ == "__main__":
    interactive_recommendations()
