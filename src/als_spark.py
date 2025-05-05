"""
Spark-based Alternating Least Squares (ALS) implementation for collaborative filtering.

This module provides a distributed implementation of the ALS algorithm using PySpark.
It's designed to handle large datasets by leveraging Spark's distributed computing capabilities.
"""

import os
import time
import psutil
from typing import Dict, Tuple, Any, Optional, List

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


class SparkALS:
    """collaborative filtering recommendation model using Spark's ALS implementation."""

    def __init__(
        self,
        spark: SparkSession,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: str = "rating",
        prediction_col: str = "prediction",
    ):
        """
        Initialize the SparkALS model
        """
        self.spark = spark
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.prediction_col = prediction_col
        self.model = None
        self.metrics = {}

        # Add tracking for memory usage
        self.memory_metrics = []

    def _track_memory_usage(self, stage: str):
        """Track memory usage at a specific stage of processing"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        # Get Spark memory metrics if available
        spark_metrics = {}
        try:
            spark_memory = (
                self.spark.sparkContext._jvm.org.apache.spark.SparkEnv.get().memoryManager()
            )
            spark_metrics["executor_memory_used"] = spark_memory.executionMemoryUsed()
            spark_metrics["storage_memory_used"] = spark_memory.storageMemoryUsed()
        except:
            # Might not be available in all Spark versions/configurations
            pass

        self.memory_metrics.append(
            {
                "stage": stage,
                "timestamp": time.time(),
                "rss_memory_mb": memory_info.rss / (1024 * 1024),
                "spark_metrics": spark_metrics,
            }
        )

        return memory_info.rss / (1024 * 1024)

    def optimize_partitioning(
        self, data: DataFrame, target_size_mb: int = 128
    ) -> DataFrame:
        # figure out how data is currently divided
        current_partitions = data.rdd.getNumPartitions()

        # keep data in memory and count rows
        data.cache()
        count = data.count()

        # estimate total size assuming each rating takes ~25 bytes
        estimated_size_mb = (count * 25) / (1024 * 1024)

        # calculate how many chunks we need based on target size
        optimal_partitions = max(1, int(estimated_size_mb / target_size_mb))

        # make sure we use all available cpu cores
        min_partitions = self.spark.sparkContext.defaultParallelism
        optimal_partitions = max(optimal_partitions, min_partitions)

        print(f"Dataset size: ~{estimated_size_mb:.2f}MB")
        print(f"Current partitions: {current_partitions}")
        print(f"Optimal partitions: {optimal_partitions}")

        # only reorganize data if it's worth the effort
        if abs(optimal_partitions - current_partitions) > current_partitions * 0.2:
            print(
                f"Repartitioning from {current_partitions} to {optimal_partitions} partitions"
            )
            return data.repartition(optimal_partitions)

        return data

    def train(
        self,
        train_data: DataFrame,
        validation_data: Optional[DataFrame] = None,
        rank: int = 10,
        max_iter: int = 10,
        reg_param: float = 0.1,
        implicit_prefs: bool = False,
        alpha: float = 1.0,
        optimize_partitions: bool = True,
        num_partitions: Optional[int] = None,
    ) -> "SparkALS":
        start_time = time.time()
        self._track_memory_usage("train_start")

        # get dataset statistics
        num_ratings = train_data.count()
        num_users = train_data.select(self.user_col).distinct().count()
        num_items = train_data.select(self.item_col).distinct().count()

        print(
            f"Training dataset: {num_ratings} ratings, {num_users} users, {num_items} items"
        )

        # Optimize partitioning if requested
        if num_partitions is not None:
            print(f"Manually setting partitions to {num_partitions}")
            train_data = train_data.repartition(num_partitions)
        elif optimize_partitions:
            train_data = self.optimize_partitioning(train_data)

        # keep dataset in memory so it doesn't have to reload it from disk each time
        train_data.cache()

        # check how much memory is used after caching
        self._track_memory_usage("after_caching")

        # create a small lookup table with movie popularity counts
        item_count_df = train_data.groupBy(self.item_col).count()
        item_count_map = {
            row[self.item_col]: row["count"] for row in item_count_df.collect()
        }

        # share item counts with all worker machines in the cluster
        self.broadcast_item_counts = self.spark.sparkContext.broadcast(item_count_map)

        # set up checkpointing for failure recovery
        checkpoint_dir = f"/tmp/spark_checkpoint_{int(time.time())}"
        self.spark.sparkContext.setCheckpointDir(checkpoint_dir)

        self._track_memory_usage("after_broadcast")

        # Initialize ALS
        als = ALS(
            rank=rank,
            maxIter=max_iter,
            regParam=reg_param,
            userCol=self.user_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
            coldStartStrategy="drop",
            implicitPrefs=implicit_prefs,
            alpha=alpha,
            nonnegative=False,
            # set number of blocks for parallel computation
            numItemBlocks=10,  # Adjust based on cluster size
            numUserBlocks=10,  # Adjust based on cluster size
            # Set checkpoint interval to enable recovery for long-running jobs
            checkpointInterval=2,
        )

        print(
            f"Training Spark ALS model with rank={rank}, maxIter={max_iter}, regParam={reg_param}"
        )

        # Train the model
        train_iter_start = time.time()
        self.model = als.fit(train_data)
        train_iter_time = time.time() - train_iter_start

        self._track_memory_usage("after_training")

        # Calculate training time
        training_time = time.time() - start_time
        self.metrics["training_time"] = training_time
        self.metrics["per_iteration_time"] = train_iter_time / max_iter

        # Store dataset statistics
        self.metrics["num_ratings"] = num_ratings
        self.metrics["num_users"] = num_users
        self.metrics["num_items"] = num_items

        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Average time per iteration: {train_iter_time/max_iter:.2f} seconds")

        # Evaluate on validation set if provided
        if validation_data is not None:
            self.evaluate(validation_data)

        return self

    def run_scalability_test(
        self,
        full_data: DataFrame,
        fractions: List[float] = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0],
        rank: int = 10,
        max_iter: int = 5,
        reg_param: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Run a scalability test by training on progressively larger data fractions.

        Args:
            full_data: Full training dataset
            fractions: List of fractions of the full dataset to test
            rank, max_iter, reg_param: Model hyperparameters

        Returns:
            Dictionary with scalability metrics
        """
        results = {"fractions": fractions, "metrics": []}

        # Cache the full dataset
        full_data.cache()
        total_count = full_data.count()
        print(f"Full dataset contains {total_count} ratings")

        for fraction in fractions:
            print(
                f"\nTesting with {fraction*100:.1f}% of data ({int(total_count*fraction)} ratings)"
            )

            # Create a sample of the data
            if fraction < 1.0:
                sample_data = full_data.sample(fraction=fraction, seed=42)
            else:
                sample_data = full_data

            # Cache this sample
            sample_data.cache()
            sample_count = sample_data.count()

            # Track memory before training
            mem_before = self._track_memory_usage(f"before_train_{fraction}")

            # Train on this sample
            start_time = time.time()
            self.train(
                train_data=sample_data,
                rank=rank,
                max_iter=max_iter,
                reg_param=reg_param,
                optimize_partitions=True,
            )
            train_time = time.time() - start_time

            # Track memory after training
            mem_after = self._track_memory_usage(f"after_train_{fraction}")

            # Record metrics
            metric = {
                "fraction": fraction,
                "sample_size": sample_count,
                "training_time": train_time,
                "time_per_rating": train_time / sample_count if sample_count > 0 else 0,
                "memory_before_mb": mem_before,
                "memory_after_mb": mem_after,
                "memory_increase_mb": mem_after - mem_before,
            }
            results["metrics"].append(metric)

            print(
                f"Fraction: {fraction}, Training time: {train_time:.2f}s, "
                f"Time per rating: {train_time/sample_count*1000:.4f}ms"
            )

            # Unpersist the sample to free memory
            sample_data.unpersist()

        # Create scalability visualization data
        sizes = [m["sample_size"] for m in results["metrics"]]
        times = [m["training_time"] for m in results["metrics"]]

        # Calculate speedup compared to linear scaling
        if len(sizes) > 1 and sizes[0] > 0 and times[0] > 0:
            linear_times = [times[0] * (size / sizes[0]) for size in sizes]
            speedups = [linear / actual for linear, actual in zip(linear_times, times)]
            results["linear_times"] = linear_times
            results["speedups"] = speedups

            print("\nSpeedup compared to linear scaling:")
            for i, speedup in enumerate(speedups):
                print(
                    f"  {fractions[i]*100:.1f}%: {speedup:.2f}x faster than linear scaling"
                )

        return results

    def tune_hyperparameters(
        self,
        train_data: DataFrame,
        validation_data: DataFrame,
        param_grid: Optional[Dict] = None,
        num_folds: int = 3,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Tune hyperparameters using cross-validation.

        Args:
            train_data: Training data as a Spark DataFrame
            validation_data: Validation data for evaluation
            param_grid: Dictionary of parameter grids to search
            num_folds: Number of folds for cross-validation

        Returns:
            Tuple of (best_params, best_metric)
        """
        if param_grid is None:
            param_grid = {
                "rank": [5, 10, 15],
                "regParam": [0.01, 0.1, 1.0],
                "maxIter": [10],
            }

        print("Starting hyperparameter tuning with cross-validation")
        start_time = time.time()

        # Initialize ALS
        als = ALS(
            userCol=self.user_col,
            itemCol=self.item_col,
            ratingCol=self.rating_col,
            coldStartStrategy="drop",
        )

        # Create parameter grid
        param_builder = ParamGridBuilder()
        for param_name, values in param_grid.items():
            param_builder = param_builder.addGrid(getattr(als, param_name), values)

        grid = param_builder.build()

        # Define evaluator
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol=self.rating_col,
            predictionCol=self.prediction_col,
        )

        # Create cross validator
        cv = CrossValidator(
            estimator=als,
            estimatorParamMaps=grid,
            evaluator=evaluator,
            numFolds=num_folds,
        )

        # Run cross-validation
        cv_model = cv.fit(train_data)

        # Extract best parameters
        best_model = cv_model.bestModel
        best_params = {
            "rank": best_model.getRank(),
            "regParam": best_model.getRegParam(),
            "maxIter": best_model.getMaxIter(),
        }

        # Save best model
        self.model = best_model

        # Evaluate on validation data
        best_metric = self.evaluate(validation_data)

        tuning_time = time.time() - start_time
        self.metrics["tuning_time"] = tuning_time

        print(f"Hyperparameter tuning completed in {tuning_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Best RMSE: {best_metric:.4f}")

        return best_params, best_metric

    def predict(self, test_data: DataFrame) -> DataFrame:
        """
        Generate predictions for the test data.

        Args:
            test_data: Test data as a Spark DataFrame

        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.transform(test_data)

    def evaluate(self, test_data: DataFrame) -> float:
        """
        Evaluate the model on test data.

        Args:
            test_data: Test data as a Spark DataFrame

        Returns:
            RMSE value
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Generate predictions
        predictions = self.predict(test_data)

        # Create evaluator
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol=self.rating_col,
            predictionCol=self.prediction_col,
        )

        # Calculate RMSE
        rmse = evaluator.evaluate(predictions)
        self.metrics["rmse"] = rmse

        # Calculate MAE
        evaluator.setMetricName("mae")
        mae = evaluator.evaluate(predictions)
        self.metrics["mae"] = mae

        print(f"Evaluation results - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        return rmse

    def recommend_for_all_users(self, num_items: int = 10) -> DataFrame:
        """
        Generate top N recommendations for all users.

        Args:
            num_items: Number of recommendations per user

        Returns:
            DataFrame with user recommendations
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        return self.model.recommendForAllUsers(num_items)

    def recommend_for_user(self, user_id: int, num_items: int = 10) -> DataFrame:
        """
        Generate top N recommendations for a specific user.

        Args:
            user_id: User ID
            num_items: Number of recommendations

        Returns:
            DataFrame with recommendations for the user
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Create a DataFrame with a single user
        user_df = self.spark.createDataFrame([(user_id,)], [self.user_col])

        # Generate recommendations
        return self.model.recommendForUserSubset(user_df, num_items)

    def recommend_items_excluding_rated(
        self, train_data: DataFrame, num_items: int = 10
    ) -> DataFrame:
        """
        Generate recommendations excluding already rated items.

        Args:
            train_data: Training data with existing ratings
            num_items: Number of recommendations per user

        Returns:
            DataFrame with recommendations excluding rated items
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Get distinct users and items
        users = train_data.select(self.user_col).distinct()
        items = train_data.select(self.item_col).distinct()

        # Cross join to get all possible user-item combinations
        user_item_pairs = users.crossJoin(items)

        # Generate predictions for all pairs
        predictions = self.model.transform(user_item_pairs)

        # Remove already rated items
        rated_items = train_data.select(self.user_col, self.item_col).withColumn(
            "rated", F.lit(1)
        )

        recommendations = predictions.join(
            rated_items,
            on=[self.user_col, self.item_col],
            how="left_anti",  # Keep only rows that don't match
        )

        # Window function to get top N for each user
        window = Window.partitionBy(self.user_col).orderBy(F.desc(self.prediction_col))

        # Get top N recommendations for each user
        top_n_recs = (
            recommendations.withColumn("rank", F.rank().over(window))
            .filter(F.col("rank") <= num_items)
            .drop("rank")
        )

        return top_n_recs

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Use overwrite to handle existing model
        self.model.write().overwrite().save(model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> "SparkALS":
        """
        Load a previously trained model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Self for method chaining
        """
        self.model = ALS.load(model_path)
        print(f"Model loaded from {model_path}")
        return self

    def get_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics

    def get_memory_metrics(self) -> List[Dict]:
        """
        Get memory usage metrics collected during processing.

        Returns:
            List of memory metrics dictionaries
        """
        return self.memory_metrics
