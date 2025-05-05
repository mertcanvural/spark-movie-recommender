"""
NumPy-based Alternating Least Squares (ALS) implementation for collaborative filtering.

This module provides a single-machine implementation of the ALS algorithm using NumPy.
It's designed for comparison with the distributed Spark implementation.
"""

import os
import time
import numpy as np
import pandas as pd

from typing import Dict, Tuple, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error


class NumPyALS:
    """Collaborative filtering recommendation model using NumPy-based ALS implementation."""

    def __init__(
        self,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: str = "rating",
    ):
        """
        Initialize the NumPyALS model
        """
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.user_factors = None
        self.item_factors = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.metrics = {}

    def _create_mappings(self, ratings: pd.DataFrame) -> Tuple[np.ndarray, int, int]:
        """
        Create mappings between user/item IDs and internal indices.

        Args:
            ratings: DataFrame with user-item ratings

        Returns:
            Tuple of (ratings matrix, num_users, num_items)
        """
        # get all unique users and items from the dataframe
        unique_users = ratings[self.user_col].unique()
        unique_items = ratings[self.item_col].unique()

        # create id-to-index mappings
        self.user_map = {user: i for i, user in enumerate(unique_users)}
        self.item_map = {item: i for i, item in enumerate(unique_items)}

        # create reverse mappings for lookups
        self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        # get matrix dimensions
        num_users = len(unique_users)
        num_items = len(unique_items)

        # create dense ratings matrix in memory
        # this loads ALL data into a single machine's memory
        ratings_matrix = np.zeros((num_users, num_items))

        # fill matrix with actual ratings
        for _, row in ratings.iterrows():
            user_idx = self.user_map[row[self.user_col]]
            item_idx = self.item_map[row[self.item_col]]
            ratings_matrix[user_idx, item_idx] = row[self.rating_col]

        return ratings_matrix, num_users, num_items

    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        rank: int = 10,
        max_iter: int = 10,
        reg_param: float = 0.1,
    ) -> "NumPyALS":
        """
        Train the ALS model using NumPy.

        Args:
            train_data: Training data as a pandas DataFrame
            validation_data: Optional validation data for evaluation during training
            rank: Number of latent factors
            max_iter: Maximum iterations
            reg_param: Regularization parameter

        Returns:
            Self for method chaining
        """
        start_time = time.time()

        print(
            f"Training NumPy ALS model with rank={rank}, maxIter={max_iter}, regParam={reg_param}"
        )

        # create ratings matrix and mappings
        ratings_matrix, num_users, num_items = self._create_mappings(train_data)

        # initialize user and item factors
        self.user_factors = np.random.normal(0, 0.1, (num_users, rank))
        self.item_factors = np.random.normal(0, 0.1, (num_items, rank))

        # precompute user and item indices with ratings
        user_indices, item_indices = ratings_matrix.nonzero()

        # track training progress
        rmse_values = []

        # ALS iterations
        for iteration in range(max_iter):
            iter_start = time.time()

            # fix item factors and solve for user factors
            for user_idx in range(num_users):
                # find items rated by this user
                item_idx = np.where(ratings_matrix[user_idx] > 0)[0]

                if len(item_idx) == 0:
                    continue

                # get ratings
                ratings = ratings_matrix[user_idx, item_idx]

                # get item factors
                item_factors_subset = self.item_factors[item_idx, :]

                # compute user factor
                A = item_factors_subset.T @ item_factors_subset + reg_param * np.eye(
                    rank
                )
                b = item_factors_subset.T @ ratings

                self.user_factors[user_idx] = np.linalg.solve(A, b)

            # fix user factors and solve for item factors
            for item_idx in range(num_items):
                # Find users who rated this item
                user_idx = np.where(ratings_matrix[:, item_idx] > 0)[0]

                if len(user_idx) == 0:
                    continue

                # Get ratings from those users
                ratings = ratings_matrix[user_idx, item_idx]

                # Get user factors for those users
                user_factors_subset = self.user_factors[user_idx, :]

                # Compute item factor
                A = user_factors_subset.T @ user_factors_subset + reg_param * np.eye(
                    rank
                )
                b = user_factors_subset.T @ ratings

                self.item_factors[item_idx] = np.linalg.solve(A, b)

            # Compute RMSE on training set
            pred = np.zeros(len(user_indices))
            for i in range(len(user_indices)):
                u, j = user_indices[i], item_indices[i]
                pred[i] = self.user_factors[u] @ self.item_factors[j]

            actual = ratings_matrix[user_indices, item_indices]
            rmse = np.sqrt(np.mean((pred - actual) ** 2))
            rmse_values.append(rmse)

            iter_time = time.time() - iter_start
            print(
                f"Iteration {iteration+1}/{max_iter}: RMSE = {rmse:.4f}, Time: {iter_time:.2f}s"
            )

        # Calculate training time
        training_time = time.time() - start_time
        self.metrics["training_time"] = training_time
        self.metrics["final_train_rmse"] = rmse_values[-1]

        print(f"Training completed in {training_time:.2f} seconds")

        # Evaluate on validation set if provided
        if validation_data is not None:
            self.evaluate(validation_data)

        return self

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for the test data.

        Args:
            test_data: Test data as a pandas DataFrame

        Returns:
            DataFrame with predictions
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained yet.")

        # Create a copy of the test data
        predictions = test_data.copy()

        # Add prediction column
        predictions["prediction"] = predictions.apply(
            lambda row: self._predict_single(row[self.user_col], row[self.item_col]),
            axis=1,
        )

        return predictions

    def _predict_single(self, user_id, item_id):
        """Predict rating for a single user-item pair."""
        # Check if user and item exist in the training data
        if user_id not in self.user_map or item_id not in self.item_map:
            return 0.0  # Cold start: return default prediction

        user_idx = self.user_map[user_id]
        item_idx = self.item_map[item_id]

        prediction = np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        return float(prediction)

    def evaluate(self, test_data: pd.DataFrame) -> float:
        """
        Evaluate the model on test data.

        Args:
            test_data: Test data as a pandas DataFrame

        Returns:
            RMSE value
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained yet.")

        # Generate predictions
        predictions = self.predict(test_data)

        # Filter out rows where prediction is possible
        valid_preds = predictions[
            predictions[self.user_col].isin(self.user_map)
            & predictions[self.item_col].isin(self.item_map)
        ]

        if len(valid_preds) == 0:
            raise ValueError("No valid predictions could be made.")

        # Calculate RMSE
        rmse = np.sqrt(
            mean_squared_error(valid_preds[self.rating_col], valid_preds["prediction"])
        )
        self.metrics["rmse"] = rmse

        # Calculate MAE
        mae = mean_absolute_error(
            valid_preds[self.rating_col], valid_preds["prediction"]
        )
        self.metrics["mae"] = mae

        print(f"Evaluation results - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        return rmse

    def recommend_for_user(self, user_id, n=10, exclude_rated=True, train_data=None):
        """
        Generate top N recommendations for a specific user.

        Args:
            user_id: User ID to generate recommendations for
            n: Number of recommendations
            exclude_rated: Whether to exclude already rated items
            train_data: Training data with existing ratings (needed if exclude_rated=True)

        Returns:
            DataFrame with recommendations
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained yet.")

        if user_id not in self.user_map:
            raise ValueError(f"User {user_id} not found in training data.")

        user_idx = self.user_map[user_id]
        user_vector = self.user_factors[user_idx]

        # Calculate scores for all items
        scores = np.dot(self.item_factors, user_vector)

        if exclude_rated and train_data is not None:
            # Get items already rated by this user
            rated_items = train_data[train_data[self.user_col] == user_id][
                self.item_col
            ].values

            # Create a mask for items that should be excluded
            mask = np.ones(len(scores), dtype=bool)
            for item in rated_items:
                if item in self.item_map:
                    mask[self.item_map[item]] = False

            # Apply mask to scores
            filtered_scores = scores.copy()
            filtered_scores[~mask] = float("-inf")
        else:
            filtered_scores = scores

        # Get top N item indices
        top_indices = np.argsort(filtered_scores)[::-1][:n]

        # Create recommendations DataFrame
        recommendations = pd.DataFrame(
            {
                self.user_col: user_id,
                self.item_col: [self.reverse_item_map[idx] for idx in top_indices],
                "prediction": filtered_scores[top_indices],
            }
        )

        return recommendations

    def recommend_for_all_users(self, n=10, exclude_rated=True, train_data=None):
        """
        Generate top N recommendations for all users.

        Args:
            n: Number of recommendations per user
            exclude_rated: Whether to exclude already rated items
            train_data: Training data with existing ratings (needed if exclude_rated=True)

        Returns:
            DataFrame with recommendations for all users
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained yet.")

        all_recommendations = []

        for user_id in self.user_map:
            try:
                user_recs = self.recommend_for_user(
                    user_id, n=n, exclude_rated=exclude_rated, train_data=train_data
                )
                all_recommendations.append(user_recs)
            except Exception as e:
                print(f"Error generating recommendations for user {user_id}: {e}")

        # Combine all recommendations
        if all_recommendations:
            return pd.concat(all_recommendations, ignore_index=True)
        else:
            return pd.DataFrame(columns=[self.user_col, self.item_col, "prediction"])

    def save_model(self, model_path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            model_path: Path to save the model
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been trained yet.")

        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)

        # Save model components
        np.save(os.path.join(model_path, "user_factors.npy"), self.user_factors)
        np.save(os.path.join(model_path, "item_factors.npy"), self.item_factors)

        # Save mappings
        np.save(os.path.join(model_path, "user_map.npy"), self.user_map)
        np.save(os.path.join(model_path, "item_map.npy"), self.item_map)

        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str) -> "NumPyALS":
        """
        Load a previously trained model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Self for method chaining
        """
        # Load model components
        self.user_factors = np.load(
            os.path.join(model_path, "user_factors.npy"), allow_pickle=True
        )
        self.item_factors = np.load(
            os.path.join(model_path, "item_factors.npy"), allow_pickle=True
        )

        # Load mappings
        self.user_map = np.load(
            os.path.join(model_path, "user_map.npy"), allow_pickle=True
        ).item()
        self.item_map = np.load(
            os.path.join(model_path, "item_map.npy"), allow_pickle=True
        ).item()

        # Create reverse mappings
        self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        self.reverse_item_map = {i: item for item, i in self.item_map.items()}

        print(f"Model loaded from {model_path}")
        return self

    def get_metrics(self) -> Dict[str, float]:
        """
        Get model performance metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics
