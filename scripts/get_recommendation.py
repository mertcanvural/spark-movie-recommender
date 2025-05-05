#!/usr/bin/env python3
"""
Simple script to get recommendations for a user ID
"""

import os
import sys
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
MODEL_PATH = "models/spark/spark_als_model"
MOVIES_PATH = "data/processed/movies.csv"


def get_recommendations(user_id, num_items=10):
    """Get recommendations for a user"""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # Load movies data
    if not os.path.exists(MOVIES_PATH):
        print(f"Error: Movies data not found at {MOVIES_PATH}")
        return

    try:
        # Load movies data
        movies_df = pd.read_csv(MOVIES_PATH)
        print(f"Loaded {len(movies_df)} movies")

        # Load model
        model = ALSModel.load(MODEL_PATH)
        print("Model loaded successfully")

        # Create a DataFrame with a single user
        user_df = spark.createDataFrame([(user_id,)], ["userId"])

        # Get recommendations
        recommendations = model.recommendForUserSubset(user_df, num_items)

        # Convert to pandas and format results
        if recommendations.count() == 0:
            print(f"No recommendations found for user {user_id}")
            return

        # Extract recommendations
        pdf = recommendations.toPandas()
        recs = pdf.iloc[0]["recommendations"]

        # Display recommendations
        print(f"\nTop {len(recs)} recommendations for user {user_id}:")
        print("-" * 80)

        for rec in recs:
            movie_id = rec["movieId"]
            score = rec["rating"]

            # Find movie details
            movie = movies_df[movies_df["movieId"] == movie_id]
            if len(movie) > 0:
                title = movie.iloc[0]["title"]
                genres = movie.iloc[0]["genres"]
                print(f"Movie: {title} (ID: {movie_id})")
                print(f"Score: {score:.2f}")
                print(f"Genres: {genres}")
                print("-" * 40)
            else:
                print(f"Movie ID: {movie_id}, Score: {score:.2f}")
                print("-" * 40)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_recommendation.py <user_id>")
        print("Example: python get_recommendation.py 1")
        sys.exit(1)

    try:
        user_id = int(sys.argv[1])
        get_recommendations(user_id)
    except ValueError:
        print("Error: User ID must be an integer")
    except KeyboardInterrupt:
        print("\nOperation cancelled")
    finally:
        spark.stop()
