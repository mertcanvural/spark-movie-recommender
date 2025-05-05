#!/usr/bin/env python3
"""
Compare Spark ALS and NumPy ALS implementations for movie recommendations.

This script trains both implementations with the same parameters and data,
then compares their performance and runtimes to showcase Spark's advantages.
"""

import os
import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import spark dependencies
from pyspark.sql import SparkSession
from pyspark import SparkConf

# Import the model implementations
from src.als_spark import SparkALS
from src.numpy_als import NumPyALS

# Suppress warnings
import warnings

warnings.filterwarnings("ignore")


def create_spark_session(app_name="MovieRecommender-SparkALS", memory="4g"):
    """Create a Spark session with appropriate configuration."""
    conf = SparkConf().setAppName(app_name)
    conf.set("spark.executor.memory", memory)
    conf.set("spark.driver.memory", memory)
    conf.set("spark.sql.shuffle.partitions", "10")

    # Create a SparkSession
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    return spark


def load_spark_data(spark, data_dir, file_format="parquet"):
    """
    Load training and test data for Spark.

    Args:
        spark: Spark session
        data_dir: Directory containing the data
        file_format: Format of the data files (parquet or csv)
    """
    if file_format == "parquet":
        # Load from Parquet files (optimized for Spark)
        train_path = os.path.join(data_dir, "spark", "train_ratings.parquet")
        test_path = os.path.join(data_dir, "spark", "test_ratings.parquet")

        train_data = spark.read.parquet(train_path)
        test_data = spark.read.parquet(test_path)
    else:
        # Load from CSV files (fallback)
        train_path = os.path.join(data_dir, "train_ratings.csv")
        test_path = os.path.join(data_dir, "test_ratings.csv")

        train_data = spark.read.csv(train_path, header=True, inferSchema=True)
        test_data = spark.read.csv(test_path, header=True, inferSchema=True)

    # Cache the data for better performance
    train_data.cache()
    test_data.cache()

    print(f"Loaded Spark training data: {train_data.count()} ratings")
    print(f"Loaded Spark test data: {test_data.count()} ratings")

    return train_data, test_data


def load_numpy_data(data_dir):
    """
    Load training and test data for NumPy.

    Args:
        data_dir: Directory containing the data
    """
    # Load from CSV files
    train_path = os.path.join(data_dir, "train_ratings.csv")
    test_path = os.path.join(data_dir, "test_ratings.csv")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print(f"Loaded NumPy training data: {len(train_data)} ratings")
    print(f"Loaded NumPy test data: {len(test_data)} ratings")

    return train_data, test_data


def train_spark_model(spark, train_data, test_data, model_dir, params):
    """
    Train the Spark ALS model and save it.

    Args:
        spark: Spark session
        train_data: Training data
        test_data: Test data
        model_dir: Directory to save the model
        params: Model hyperparameters
    """
    print("\n" + "=" * 50)
    print("TRAINING SPARK ALS MODEL")
    print("=" * 50)

    # Start timing
    start_time = time.time()

    # Initialize model
    model = SparkALS(spark)

    # Train the model
    model.train(
        train_data=train_data,
        validation_data=test_data,
        rank=params["rank"],
        max_iter=params["max_iter"],
        reg_param=params["reg_param"],
        implicit_prefs=params["implicit_prefs"],
    )

    # Evaluate the model and store the result
    evaluation_rmse = model.evaluate(test_data)
    print(f"Spark ALS Evaluation RMSE: {evaluation_rmse}")

    # Calculate total time
    total_time = time.time() - start_time

    # Save the model
    model_path = os.path.join(model_dir, "spark", "spark_als_model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)

    # Save metrics
    metrics = model.get_metrics()
    metrics["total_time"] = total_time
    metrics_path = os.path.join(model_dir, "spark", "metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Spark ALS model saved to {model_path}")
    print(f"Spark ALS metrics saved to {metrics_path}")
    print(f"Spark ALS total execution time: {total_time:.2f} seconds")

    return model, metrics, total_time


def train_numpy_model(train_data, test_data, model_dir, params):
    """
    Train the NumPy ALS model and save it.

    Args:
        train_data: Training data
        test_data: Test data
        model_dir: Directory to save the model
        params: Model hyperparameters
    """
    print("\n" + "=" * 50)
    print("TRAINING NUMPY ALS MODEL")
    print("=" * 50)

    # Start timing
    start_time = time.time()

    # Initialize model
    model = NumPyALS()

    # Train the model
    model.train(
        train_data=train_data,
        validation_data=test_data,
        rank=params["rank"],
        max_iter=params["max_iter"],
        reg_param=params["reg_param"],
    )

    # Evaluate the model and store the result
    evaluation_rmse = model.evaluate(test_data)
    print(f"NumPy ALS Evaluation RMSE: {evaluation_rmse}")

    # Calculate total time
    total_time = time.time() - start_time

    # Save the model
    model_path = os.path.join(model_dir, "numpy", "numpy_als_model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)

    # Save metrics
    metrics = model.get_metrics()
    metrics["total_time"] = total_time
    metrics_path = os.path.join(model_dir, "numpy", "metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"NumPy ALS model saved to {model_path}")
    print(f"NumPy ALS metrics saved to {metrics_path}")
    print(f"NumPy ALS total execution time: {total_time:.2f} seconds")

    return model, metrics, total_time


def generate_spark_recommendations(model, train_data, model_dir):
    """
    Generate and save recommendations using Spark ALS model.

    Args:
        model: Trained SparkALS model
        train_data: Training data
        model_dir: Directory to save recommendations
    """
    # Generate recommendations excluding already rated items
    recommendations = model.recommend_items_excluding_rated(
        train_data=train_data, num_items=10
    )

    # Save recommendations
    recs_path = os.path.join(model_dir, "spark", "recommendations.parquet")
    recommendations.write.parquet(recs_path, mode="overwrite")

    # Also save as CSV for easy viewing
    csv_path = os.path.join(model_dir, "spark", "recommendations.csv")
    recommendations.toPandas().to_csv(csv_path, index=False)

    print(f"Generated {recommendations.count()} Spark recommendations")
    print(f"Spark recommendations saved to {csv_path}")


def generate_numpy_recommendations(model, train_data, model_dir):
    """
    Generate and save recommendations using NumPy ALS model.

    Args:
        model: Trained NumPyALS model
        train_data: Training data
        model_dir: Directory to save recommendations
    """
    # Generate recommendations
    recommendations = model.recommend_for_all_users(
        n=10, exclude_rated=True, train_data=train_data
    )

    # Save as CSV
    csv_path = os.path.join(model_dir, "numpy", "recommendations.csv")
    recommendations.to_csv(csv_path, index=False)

    print(f"Generated {len(recommendations)} NumPy recommendations")
    print(f"NumPy recommendations saved to {csv_path}")


def compare_performance(spark_time, numpy_time, spark_metrics, numpy_metrics):
    """
    Compare and display performance metrics between Spark and NumPy implementations.

    Args:
        spark_time: Total execution time for Spark ALS
        numpy_time: Total execution time for NumPy ALS
        spark_metrics: Metrics from Spark ALS model
        numpy_metrics: Metrics from NumPy ALS model
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: SPARK ALS vs NUMPY ALS")
    print("=" * 60)

    # Calculate speedup
    speedup = numpy_time / spark_time if spark_time > 0 else float("inf")

    # Display comparison
    print(f"{'Metric':<20} {'Spark ALS':<15} {'NumPy ALS':<15} {'Difference':<15}")
    print("-" * 60)
    print(
        f"{'Total Time (s)':<20} {spark_time:<15.2f} {numpy_time:<15.2f} {speedup:>6.2f}x faster"
    )

    if "rmse" in spark_metrics and "rmse" in numpy_metrics:
        spark_rmse = spark_metrics["rmse"]
        numpy_rmse = numpy_metrics["rmse"]
        rmse_diff = abs(spark_rmse - numpy_rmse)
        print(
            f"{'RMSE':<20} {spark_rmse:<15.4f} {numpy_rmse:<15.4f} {rmse_diff:<15.4f}"
        )

    if "mae" in spark_metrics and "mae" in numpy_metrics:
        spark_mae = spark_metrics["mae"]
        numpy_mae = numpy_metrics["mae"]
        mae_diff = abs(spark_mae - numpy_mae)
        print(f"{'MAE':<20} {spark_mae:<15.4f} {numpy_mae:<15.4f} {mae_diff:<15.4f}")

    if "training_time" in spark_metrics and "training_time" in numpy_metrics:
        spark_train = spark_metrics["training_time"]
        numpy_train = numpy_metrics["training_time"]
        train_speedup = numpy_train / spark_train if spark_train > 0 else float("inf")
        print(
            f"{'Training Time (s)':<20} {spark_train:<15.2f} {numpy_train:<15.2f} {train_speedup:>6.2f}x faster"
        )

    print("\nConclusion:")
    if speedup > 1:
        print(f"Spark ALS was {speedup:.2f}x faster than NumPy ALS overall.")
        print(
            "This demonstrates the advantage of Spark's distributed computing for large datasets."
        )
    elif speedup < 1:
        print(f"NumPy ALS was {1/speedup:.2f}x faster than Spark ALS overall.")
        print(
            "This suggests the dataset may be too small to benefit from Spark's distributed computing,"
        )
        print(
            "or the overhead of Spark initialization exceeded the computational advantage."
        )
    else:
        print("Both implementations had similar performance.")


def main():
    """Main function to train and compare ALS models."""
    parser = argparse.ArgumentParser(description="Compare Spark and NumPy ALS models")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/interim",
        help="Directory containing the data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save the models",
    )
    parser.add_argument(
        "--file-format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Format of the data files for Spark",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=10,
        help="Number of latent factors",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="Maximum number of iterations",
    )
    parser.add_argument(
        "--reg-param",
        type=float,
        default=0.05,
        help="Regularization parameter",
    )
    parser.add_argument(
        "--implicit",
        action="store_true",
        help="Use implicit preference model",
    )

    args = parser.parse_args()

    # Set common parameters for both models
    params = {
        "rank": args.rank,
        "max_iter": args.max_iter,
        "reg_param": args.reg_param,
        "implicit_prefs": args.implicit,
    }

    # Create Spark session
    spark = create_spark_session()

    try:
        # Load data for both models
        spark_train_data, spark_test_data = load_spark_data(
            spark=spark, data_dir=args.data_dir, file_format=args.file_format
        )
        numpy_train_data, numpy_test_data = load_numpy_data(args.data_dir)

        # Train Spark model
        spark_model, spark_metrics, spark_time = train_spark_model(
            spark=spark,
            train_data=spark_train_data,
            test_data=spark_test_data,
            model_dir=args.model_dir,
            params=params,
        )

        # Generate Spark recommendations
        generate_spark_recommendations(
            model=spark_model,
            train_data=spark_train_data,
            model_dir=args.model_dir,
        )

        # Train NumPy model
        numpy_model, numpy_metrics, numpy_time = train_numpy_model(
            train_data=numpy_train_data,
            test_data=numpy_test_data,
            model_dir=args.model_dir,
            params=params,
        )

        # Generate NumPy recommendations
        generate_numpy_recommendations(
            model=numpy_model,
            train_data=numpy_train_data,
            model_dir=args.model_dir,
        )

        # Compare performance
        compare_performance(
            spark_time=spark_time,
            numpy_time=numpy_time,
            spark_metrics=spark_metrics,
            numpy_metrics=numpy_metrics,
        )

    finally:
        # Clean up
        spark.stop()
        print("\nSpark session stopped")


if __name__ == "__main__":
    main()
