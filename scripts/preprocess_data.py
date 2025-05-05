#!/usr/bin/env python3
"""
Preprocess the MovieLens dataset for use in the recommendation system.
This script converts the raw MovieLens data into a format suitable for both
the Spark-based and non-Spark implementations of the ALS algorithm.

By default, this script processes all available dataset sizes that have been
downloaded to the raw data directory.
"""

import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_movielens_100k(data_dir):
    """
    Load the MovieLens 100K dataset.

    Args:
        data_dir: Path to the extracted MovieLens 100K data directory

    Returns:
        tuple: (ratings_df, movies_df, users_df)
    """
    print("Loading MovieLens 100K dataset...")

    # load ratings
    ratings_path = os.path.join(data_dir, "u.data")
    ratings_df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    # load movie information
    movies_path = os.path.join(data_dir, "u.item")
    movies_df = pd.read_csv(
        movies_path,
        sep="|",
        names=["movieId", "title", "release_date", "video_release_date", "IMDb_URL"]
        + [f"genre_{i}" for i in range(19)],
        encoding="latin-1",
    )

    # load user information
    users_path = os.path.join(data_dir, "u.user")
    users_df = pd.read_csv(
        users_path,
        sep="|",
        names=["userId", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1",
    )

    return ratings_df, movies_df, users_df


def load_movielens_1m(data_dir):
    """
    Load the MovieLens 1M dataset.

    Args:
        data_dir: Path to the extracted MovieLens 1M data directory

    Returns:
        tuple: (ratings_df, movies_df, users_df)
    """
    print("Loading MovieLens 1M dataset...")

    # load ratings
    ratings_path = os.path.join(data_dir, "ratings.dat")
    ratings_df = pd.read_csv(
        ratings_path,
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",  # Needed for the custom separator
        encoding="latin-1",
    )

    # load movie information
    movies_path = os.path.join(data_dir, "movies.dat")
    movies_df = pd.read_csv(
        movies_path,
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    # load user information
    users_path = os.path.join(data_dir, "users.dat")
    users_df = pd.read_csv(
        users_path,
        sep="::",
        names=["userId", "gender", "age", "occupation", "zip_code"],
        engine="python",
        encoding="latin-1",
    )

    return ratings_df, movies_df, users_df


def load_movielens_10m(data_dir):
    """
    Load the MovieLens 10M dataset.

    Args:
        data_dir: Path to the extracted MovieLens 10M data directory

    Returns:
        tuple: (ratings_df, movies_df, users_df)
    """
    print("Loading MovieLens 10M dataset...")

    # Check for both possible folder names
    if os.path.exists(os.path.join(data_dir, "ratings.dat")):
        ratings_path = os.path.join(data_dir, "ratings.dat")
    else:
        # Sometimes dataset is in ml-10M100K subfolder
        ratings_path = os.path.join(data_dir, "ratings.dat")
        if not os.path.exists(ratings_path):
            alt_dir = os.path.join(data_dir.parent, "ml-10M100K")
            if os.path.exists(alt_dir):
                ratings_path = os.path.join(alt_dir, "ratings.dat")

    # load ratings
    ratings_df = pd.read_csv(
        ratings_path,
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    # load movie information - check both possible locations
    movies_path = os.path.join(data_dir, "movies.dat")
    if not os.path.exists(movies_path):
        alt_dir = os.path.join(data_dir.parent, "ml-10M100K")
        if os.path.exists(alt_dir):
            movies_path = os.path.join(alt_dir, "movies.dat")

    movies_df = pd.read_csv(
        movies_path,
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    # 10M dataset doesn't have users info, create minimal DataFrame
    unique_users = ratings_df["userId"].unique()
    users_df = pd.DataFrame(
        {
            "userId": unique_users,
            "gender": "unknown",
            "age": -1,
        }
    )

    return ratings_df, movies_df, users_df


def load_movielens_25m(data_dir):
    """
    Load the MovieLens 25M dataset.

    Args:
        data_dir: Path to the extracted MovieLens 25M data directory

    Returns:
        tuple: (ratings_df, movies_df, users_df)
    """
    print("Loading MovieLens 25M dataset...")

    # load ratings - 25M dataset uses CSV format with header
    ratings_path = os.path.join(data_dir, "ratings.csv")
    ratings_df = pd.read_csv(ratings_path)

    # load movie information
    movies_path = os.path.join(data_dir, "movies.csv")
    movies_df = pd.read_csv(movies_path)

    # 25M dataset doesn't have users info, create minimal DataFrame
    unique_users = ratings_df["userId"].unique()
    users_df = pd.DataFrame(
        {
            "userId": unique_users,
            "gender": "unknown",
            "age": -1,
        }
    )

    return ratings_df, movies_df, users_df


def clean_data(ratings_df, movies_df, users_df, interim_dir):
    """
    Clean the datasets and save them to the interim directory.

    Args:
        ratings_df: DataFrame with user ratings
        movies_df: DataFrame with movie information
        users_df: DataFrame with user information
        interim_dir: Directory to save the cleaned data

    Returns:
        tuple: (cleaned_ratings_df, cleaned_movies_df, cleaned_users_df)
    """
    print("Cleaning data...")

    # ensure interim directory exists
    os.makedirs(interim_dir, exist_ok=True)

    # clean movies dataframe
    if "title" in movies_df.columns and "genres" in movies_df.columns:
        movies_cleaned = movies_df[["movieId", "title", "genres"]]
    else:
        # For 100K dataset, combine genre columns
        genre_names = [
            "Action",
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Crime",
            "Documentary",
            "Drama",
            "Fantasy",
            "Film-Noir",
            "Horror",
            "Musical",
            "Mystery",
            "Romance",
            "Sci-Fi",
            "Thriller",
            "War",
            "Western",
            "Unknown",
        ]

        # Create genres string
        def get_genres(row):
            genres = []
            for i, genre in enumerate(genre_names):
                if row.get(f"genre_{i}") == 1:
                    genres.append(genre)
            return "|".join(genres)

        movies_df["genres"] = movies_df.apply(get_genres, axis=1)
        movies_cleaned = movies_df[["movieId", "title", "genres"]]

    # Clean ratings - no additional processing needed
    ratings_cleaned = ratings_df

    # Clean users - simplify to userId, gender, age if available
    if "gender" in users_df.columns and "age" in users_df.columns:
        users_cleaned = users_df[["userId", "gender", "age"]]
    else:
        # For datasets without user info
        users_cleaned = users_df

    # Save cleaned data to interim directory
    cleaned_ratings_path = os.path.join(interim_dir, "cleaned_ratings.csv")
    cleaned_movies_path = os.path.join(interim_dir, "cleaned_movies.csv")
    cleaned_users_path = os.path.join(interim_dir, "cleaned_users.csv")

    ratings_cleaned.to_csv(cleaned_ratings_path, index=False)
    movies_cleaned.to_csv(cleaned_movies_path, index=False)
    users_cleaned.to_csv(cleaned_users_path, index=False)

    print(f"Cleaned data saved to {interim_dir}")

    return ratings_cleaned, movies_cleaned, users_cleaned


def split_data(ratings_df, interim_dir, test_size=0.2, random_state=42):
    """
    Split ratings data into training and test sets.

    Args:
        ratings_df: DataFrame with user ratings
        interim_dir: Directory to save the split data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_df, test_df)
    """
    print(f"Splitting data with test_size={test_size}...")

    # Split the data
    train_df, test_df = train_test_split(
        ratings_df, test_size=test_size, random_state=random_state
    )

    # Save split data to interim directory
    train_path = os.path.join(interim_dir, "train_ratings.csv")
    test_path = os.path.join(interim_dir, "test_ratings.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train set: {len(train_df)} samples, saved to {train_path}")
    print(f"Test set: {len(test_df)} samples, saved to {test_path}")

    return train_df, test_df


def convert_to_parquet(ratings_df, movies_df, users_df, train_df, test_df, interim_dir):
    """
    Convert dataframes to Parquet format for Spark efficiency.

    Args:
        ratings_df: Full ratings DataFrame
        movies_df: Movies DataFrame
        users_df: Users DataFrame
        train_df: Training ratings DataFrame
        test_df: Test ratings DataFrame
        interim_dir: Base interim directory
    """
    print("Converting data to Parquet format for Spark...")

    # Create spark directory if it doesn't exist
    spark_dir = os.path.join(interim_dir, "spark")
    os.makedirs(spark_dir, exist_ok=True)

    # Convert each dataframe to parquet
    ratings_df.to_parquet(os.path.join(spark_dir, "ratings.parquet"))
    movies_df.to_parquet(os.path.join(spark_dir, "movies.parquet"))
    users_df.to_parquet(os.path.join(spark_dir, "users.parquet"))
    train_df.to_parquet(os.path.join(spark_dir, "train_ratings.parquet"))
    test_df.to_parquet(os.path.join(spark_dir, "test_ratings.parquet"))

    print(f"Parquet files saved to {spark_dir}")


def convert_to_numpy(ratings_df, movies_df, users_df, train_df, test_df, interim_dir):
    """
    Convert dataframes to NumPy arrays for non-distributed implementation.

    Args:
        ratings_df: Full ratings DataFrame
        movies_df: Movies DataFrame
        users_df: Users DataFrame
        train_df: Training ratings DataFrame
        test_df: Test ratings DataFrame
        interim_dir: Base interim directory
    """
    print("Converting data to NumPy arrays...")

    # Create numpy directory if it doesn't exist
    numpy_dir = os.path.join(interim_dir, "numpy")
    os.makedirs(numpy_dir, exist_ok=True)

    # Convert ratings to arrays
    ratings_array = ratings_df[["userId", "movieId", "rating"]].values
    train_array = train_df[["userId", "movieId", "rating"]].values
    test_array = test_df[["userId", "movieId", "rating"]].values

    # Save arrays
    np.save(os.path.join(numpy_dir, "ratings.npy"), ratings_array)
    np.save(os.path.join(numpy_dir, "train_ratings.npy"), train_array)
    np.save(os.path.join(numpy_dir, "test_ratings.npy"), test_array)

    # Save movie and user mapping (for interpretation)
    movies_df[["movieId", "title"]].to_csv(
        os.path.join(numpy_dir, "movie_mapping.csv"), index=False
    )
    users_df[["userId"]].to_csv(
        os.path.join(numpy_dir, "user_mapping.csv"), index=False
    )

    print(f"NumPy arrays saved to {numpy_dir}")


def create_processed_data(train_df, test_df, movies_df, users_df, processed_dir):
    """
    Create final processed data files in a format ready for model training.

    Args:
        train_df: Training ratings DataFrame
        test_df: Test ratings DataFrame
        movies_df: Movies DataFrame
        users_df: Users DataFrame
        processed_dir: Directory to save the processed data
    """
    print("Creating final processed data...")

    # Ensure processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Save final processed data
    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    movies_df.to_csv(os.path.join(processed_dir, "movies.csv"), index=False)
    users_df.to_csv(os.path.join(processed_dir, "users.csv"), index=False)

    print(f"Final processed data saved to {processed_dir}")

    # Print some statistics
    print(f"Total users: {users_df['userId'].nunique()}")
    print(f"Total movies: {movies_df['movieId'].nunique()}")
    print(f"Total ratings: {len(train_df) + len(test_df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Average rating: {train_df['rating'].mean():.2f}")


def process_size(size, raw_data_dir, interim_dir, processed_dir, test_size=0.2):
    """Process a specific dataset size"""
    print(f"\n{'='*80}")
    print(f"PROCESSING {size.upper()} DATASET")
    print(f"{'='*80}")

    # Create size-specific directories
    size_interim_dir = os.path.join(interim_dir, size)
    size_processed_dir = os.path.join(processed_dir, size)

    # Ensure directories exist
    os.makedirs(size_interim_dir, exist_ok=True)
    os.makedirs(size_processed_dir, exist_ok=True)

    # Check if dataset is already processed
    spark_dir = os.path.join(size_processed_dir, "spark")
    train_parquet = os.path.join(spark_dir, "train_ratings.parquet")
    if os.path.exists(train_parquet):
        print(f"Dataset {size} is already processed. Skipping.")
        return True

    # Load data based on dataset size - handle special cases for folder names
    if size == "10m" and not os.path.exists(os.path.join(raw_data_dir, f"ml-{size}")):
        # Check for the alternative 10M folder name
        if os.path.exists(os.path.join(raw_data_dir, "ml-10M100K")):
            data_path = os.path.join(raw_data_dir, "ml-10M100K")
        else:
            print(
                f"Dataset {size} not found at {os.path.join(raw_data_dir, f'ml-{size}')} or ml-10M100K. Skipping."
            )
            return False
    else:
        data_path = os.path.join(raw_data_dir, f"ml-{size}")
        if not os.path.exists(data_path):
            print(f"Dataset {size} not found at {data_path}. Skipping.")
            return False

    try:
        if size == "100k":
            ratings_df, movies_df, users_df = load_movielens_100k(data_path)
        elif size == "1m":
            ratings_df, movies_df, users_df = load_movielens_1m(data_path)
        elif size == "10m":
            ratings_df, movies_df, users_df = load_movielens_10m(data_path)
        elif size == "25m":
            ratings_df, movies_df, users_df = load_movielens_25m(data_path)
        else:
            print(f"Unsupported dataset size: {size}")
            return False

        # Step 1: Clean data
        ratings_cleaned, movies_cleaned, users_cleaned = clean_data(
            ratings_df, movies_df, users_df, size_interim_dir
        )

        # Step 2: Split data into training and test sets
        train_df, test_df = split_data(
            ratings_cleaned, size_interim_dir, test_size=test_size
        )

        # Step 3: Convert to Parquet for Spark
        convert_to_parquet(
            ratings_cleaned,
            movies_cleaned,
            users_cleaned,
            train_df,
            test_df,
            size_interim_dir,
        )

        # Step 4: Convert to NumPy arrays for non-distributed implementation
        convert_to_numpy(
            ratings_cleaned,
            movies_cleaned,
            users_cleaned,
            train_df,
            test_df,
            size_interim_dir,
        )

        # Step 5: Create final processed data
        create_processed_data(
            train_df, test_df, movies_cleaned, users_cleaned, size_processed_dir
        )

        # Copy processed data to spark and numpy-specific directories
        spark_processed_dir = os.path.join(size_processed_dir, "spark")
        numpy_processed_dir = os.path.join(size_processed_dir, "numpy")

        os.makedirs(spark_processed_dir, exist_ok=True)
        os.makedirs(numpy_processed_dir, exist_ok=True)

        # Copy parquet files for Spark
        spark_interim_dir = os.path.join(size_interim_dir, "spark")
        if os.path.exists(spark_interim_dir):
            import shutil

            for file in os.listdir(spark_interim_dir):
                if file.endswith(".parquet"):
                    src = os.path.join(spark_interim_dir, file)
                    dst = os.path.join(spark_processed_dir, file)
                    shutil.copy2(src, dst)

        print(f"Processing complete for {size} dataset!")
        return True

    except Exception as e:
        print(f"Error processing {size} dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Preprocess MovieLens dataset")
    parser.add_argument(
        "--size",
        choices=["100k", "1m", "10m", "25m", "all"],
        default="all",
        help="MovieLens dataset size to process, or 'all' for all available datasets",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing",
    )
    args = parser.parse_args()

    # set up directories
    raw_data_dir = Path("data/raw")
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")

    # Ensure directories exist
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.size == "all":
        # Process all available dataset sizes
        available_sizes = []
        for size in ["100k", "1m", "10m", "25m"]:
            if os.path.exists(os.path.join(raw_data_dir, f"ml-{size}")):
                available_sizes.append(size)

        if not available_sizes:
            print("No datasets found in data/raw directory.")
            print("Please run scripts/download_data.py first.")
            return

        print(f"Found {len(available_sizes)} dataset(s): {', '.join(available_sizes)}")

        # Process each available size
        success_count = 0
        for size in available_sizes:
            if process_size(
                size, raw_data_dir, interim_dir, processed_dir, args.test_size
            ):
                success_count += 1

        print(
            f"\nSuccessfully processed {success_count} of {len(available_sizes)} datasets"
        )
    else:
        # Process specific size
        if process_size(
            args.size, raw_data_dir, interim_dir, processed_dir, args.test_size
        ):
            print(f"\nSuccessfully processed {args.size} dataset")
        else:
            print(f"\nFailed to process {args.size} dataset")


if __name__ == "__main__":
    main()
