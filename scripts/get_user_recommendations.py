#!/usr/bin/env python3
"""
Interactive script to get movie recommendations by entering user IDs.
Loads the trained Spark ALS model and provides recommendations for specific users.
"""

import os
import sys
import pandas as pd
from pyspark.sql import SparkSession

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.als_spark import SparkALS

# Initialize Spark session
spark = (
    SparkSession.builder.appName("Movie Recommendations")
    .config("spark.executor.memory", "4g")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

# Paths
MODEL_PATH = "models/spark/spark_als_model"
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
        # Initialize SparkALS and load model
        model = SparkALS(spark=spark)
        model.load_model(MODEL_PATH)
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


def format_recommendations(recs_df, movies_df):
    """Format the recommendations into a readable format"""
    # Convert PySpark DataFrame to Pandas
    pdf = recs_df.toPandas()

    if len(pdf) == 0:
        return "No recommendations found."

    # Extract recommendations
    recommendations = pdf.iloc[0]["recommendations"]

    # Format results
    results = []
    for rec in recommendations:
        movie_id = rec["movieId"]
        score = rec["rating"]

        # Look up movie details
        movie = movies_df[movies_df["movieId"] == movie_id]
        if len(movie) > 0:
            title = movie.iloc[0]["title"]
            genres = movie.iloc[0]["genres"]
            results.append(
                f"Movie ID: {movie_id}, Title: {title}, Score: {score:.2f}, Genres: {genres}"
            )
        else:
            results.append(
                f"Movie ID: {movie_id}, Score: {score:.2f}, (Movie details not found)"
            )

    return "\n".join(results)


def interactive_recommendations():
    """Interactive loop for getting recommendations"""
    print("Loading model and movie data...")
    model = load_model()
    movies_df = load_movies()

    print("\n" + "=" * 80)
    print("User-Based Movie Recommendation System")
    print("=" * 80)
    print("Enter a user ID to get recommendations, or 'q' to quit.")

    while True:
        try:
            user_input = input("\nEnter user ID (or 'q' to quit): ")

            if user_input.lower() == "q":
                break

            try:
                user_id = int(user_input)
            except ValueError:
                print("Please enter a valid user ID (integer).")
                continue

            # Get recommendations
            print(f"\nGetting recommendations for user ID: {user_id}")
            try:
                recommendations = model.recommend_for_user(
                    user_id=user_id, num_items=10
                )

                print("\nRecommended Movies:")
                print("-" * 80)
                print(format_recommendations(recommendations, movies_df))

            except Exception as e:
                print(f"Error getting recommendations: {str(e)}")
                print("This user ID may not exist in the training data.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")

    print("\nThank you for using the User-Based Movie Recommendation System!")
    spark.stop()


if __name__ == "__main__":
    interactive_recommendations()
