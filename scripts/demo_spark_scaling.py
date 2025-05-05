#!/usr/bin/env python3
"""
Spark Scaling Demo for ALS Recommendation

This script demonstrates how Spark's distributed computation scales with dataset size.
It loads different sized datasets and shows how Spark efficiently:
1. Partitions data across nodes
2. Caches data in memory
3. Distributes computation for ALS matrix factorization
4. Scales sublinearly with dataset size

Usage:
  python scripts/demo_spark_scaling.py --size 10m
"""

import os
import sys
import time
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

# Import Spark dependencies
from pyspark.sql import SparkSession, DataFrame
from pyspark import SparkConf
import pandas as pd

# Add project root to Python path
sys.path.insert(0, os.path.abspath("."))
from src.als_spark import SparkALS


def create_spark_session(memory="4g", cores=2):
    """Create a Spark session with specified configuration."""
    print("Initializing Spark session...")
    conf = SparkConf().setAppName("SparkScalingDemo")

    # Memory settings
    conf.set("spark.executor.memory", memory)
    conf.set("spark.driver.memory", memory)

    # Control the parallelism
    conf.set("spark.default.parallelism", str(cores * 2))
    conf.set("spark.sql.shuffle.partitions", str(cores * 2))

    # Disable event logging to avoid the error
    conf.set("spark.eventLog.enabled", "false")

    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    return spark


def load_dataset(spark, size, data_dir="data/processed"):
    """Load preprocessed dataset by size."""
    path = Path(data_dir) / size / "spark" / "train_ratings.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset {size} not found at {path}. Please run the pipeline first "
            f"with: python run_pipeline.py --sizes {size} --skip-numpy"
        )

    print(f"Loading {size} dataset from {path}")
    return spark.read.parquet(str(path))


def demo_distributed_advantages(spark, data, demo_type="partitioning"):
    """
    Demonstrate specific distributed computing advantages.

    Args:
        spark: SparkSession
        data: Spark DataFrame
        demo_type: Type of demo to run (partitioning, broadcast, caching, scaling)
    """
    if demo_type == "partitioning":
        # Show how data is partitioned across workers
        num_partitions = data.rdd.getNumPartitions()
        print(f"\n=== DATA PARTITIONING DEMO ===")
        print(f"Dataset has {num_partitions} partitions")

        # Count records per partition
        def count_in_partition(iterator):
            count = 0
            for _ in iterator:
                count += 1
            yield count

        partition_counts = data.rdd.mapPartitions(count_in_partition).collect()

        print("Records per partition:")
        for i, count in enumerate(partition_counts):
            print(f"  Partition {i}: {count:,} records")

        # Test with different partition counts
        partition_times = []
        partition_counts = [1, 2, 4, 8, 16]

        for partitions in partition_counts:
            start_time = time.time()
            # Force repartitioning and count action
            repartitioned = data.repartition(partitions)
            count = repartitioned.count()
            duration = time.time() - start_time

            partition_times.append(duration)
            print(
                f"Count with {partitions} partitions: {count:,} records in {duration:.2f}s"
            )

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(partition_counts, partition_times, "o-")
        plt.xlabel("Number of Partitions")
        plt.ylabel("Time (seconds)")
        plt.title("Effect of Partitioning on Data Processing Speed")
        plt.grid(True, alpha=0.3)
        plt.savefig("partition_scaling.png")
        print(f"Saved partition scaling visualization to partition_scaling.png")

    elif demo_type == "broadcast":
        # Show how broadcast variables reduce data transfer
        print(f"\n=== BROADCAST VARIABLE DEMO ===")

        # Create a lookup table of movie ratings count
        item_counts = data.groupBy("movieId").count()

        # Method 1: Without broadcast (slow for repeated access)
        start_time = time.time()
        # Convert to dictionary the slow way (collect all to driver)
        movie_count_map = {
            row["movieId"]: row["count"] for row in item_counts.collect()
        }
        driver_time = time.time() - start_time
        print(f"Time to create lookup on driver: {driver_time:.2f}s")

        # Method 2: With broadcast (faster for repeated access)
        start_time = time.time()
        # Broadcast the dictionary to all workers
        broadcast_counts = spark.sparkContext.broadcast(movie_count_map)
        broadcast_time = time.time() - start_time
        print(f"Time to broadcast lookup table: {broadcast_time:.2f}s")

        # Test accessing the data 1000 times
        def access_normal(iterator, lookup):
            count = 0
            for row in iterator:
                # Access the local copy each time
                if row["movieId"] in lookup:
                    count += lookup[row["movieId"]]
            yield count

        def access_broadcast(iterator, broadcast_lookup):
            count = 0
            # Get value once from broadcast
            lookup = broadcast_lookup.value
            for row in iterator:
                if row["movieId"] in lookup:
                    count += lookup[row["movieId"]]
            yield count

        # Time normal access
        start_time = time.time()
        normal_result = data.rdd.mapPartitions(
            lambda x: access_normal(x, movie_count_map)
        ).collect()
        normal_time = time.time() - start_time

        # Time broadcast access
        start_time = time.time()
        broadcast_result = data.rdd.mapPartitions(
            lambda x: access_broadcast(x, broadcast_counts)
        ).collect()
        broadcast_access_time = time.time() - start_time

        print(f"Access time without broadcast: {normal_time:.2f}s")
        print(f"Access time with broadcast: {broadcast_access_time:.2f}s")
        print(
            f"Speedup: {normal_time / broadcast_access_time:.2f}x faster with broadcast"
        )

    elif demo_type == "caching":
        # Show advantage of in-memory caching
        print(f"\n=== IN-MEMORY CACHING DEMO ===")

        # Force garbage collection and unpersist any cached data
        spark.catalog.clearCache()

        # First run without caching
        start_time = time.time()
        # Run 3 operations that each scan the dataset
        count1 = data.count()
        avg_rating = data.agg({"rating": "avg"}).collect()[0][0]
        unique_users = data.select("userId").distinct().count()
        uncached_time = time.time() - start_time

        print(f"Without caching: {uncached_time:.2f}s")
        print(
            f"  Count: {count1:,}, Avg rating: {avg_rating:.2f}, Users: {unique_users:,}"
        )

        # Now cache and run the same operations
        data.cache()
        # Force caching by running an action
        data.count()

        start_time = time.time()
        # Same 3 operations on cached data
        count2 = data.count()
        avg_rating = data.agg({"rating": "avg"}).collect()[0][0]
        unique_users = data.select("userId").distinct().count()
        cached_time = time.time() - start_time

        print(f"With caching: {cached_time:.2f}s")
        print(
            f"  Count: {count2:,}, Avg rating: {avg_rating:.2f}, Users: {unique_users:,}"
        )
        print(f"Speedup: {uncached_time / cached_time:.2f}x faster with caching")

    elif demo_type == "scaling":
        # Show how Spark scales with dataset size
        print(f"\n=== SCALING WITH DATASET SIZE DEMO ===")

        # Create ALS model
        als = SparkALS(spark)

        # Test with different dataset fractions
        fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
        results = als.run_scalability_test(
            full_data=data,
            fractions=fractions,
            rank=10,
            max_iter=3,  # Just a few iterations for demo
            reg_param=0.1,
        )

        # Print memory metrics
        memory_metrics = als.get_memory_metrics()
        stages = [m["stage"] for m in memory_metrics]
        memory_usage = [m["rss_memory_mb"] for m in memory_metrics]

        print("\nMemory usage during processing:")
        for stage, mem in zip(stages, memory_usage):
            print(f"  {stage}: {mem:.1f} MB")


def main():
    """Parse arguments and run the demo."""
    parser = argparse.ArgumentParser(
        description="Demonstrate Spark's distributed computing advantages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--size",
        type=str,
        default="1m",
        choices=["1m", "10m", "25m"],
        help="Dataset size to use",
    )

    parser.add_argument(
        "--demo",
        type=str,
        default="all",
        choices=["partitioning", "broadcast", "caching", "scaling", "all"],
        help="Type of demo to run",
    )

    parser.add_argument(
        "--memory", type=str, default="4g", help="Memory to allocate for Spark"
    )

    parser.add_argument("--cores", type=int, default=2, help="Number of cores to use")

    args = parser.parse_args()

    # Create Spark session
    spark = create_spark_session(memory=args.memory, cores=args.cores)

    try:
        # Load dataset
        data = load_dataset(spark, args.size)

        if args.demo == "all":
            # Run all demos
            for demo in ["partitioning", "broadcast", "caching", "scaling"]:
                demo_distributed_advantages(spark, data, demo)
        else:
            # Run specific demo
            demo_distributed_advantages(spark, data, args.demo)

    finally:
        # Stop Spark session
        spark.stop()
        print("Spark session stopped")


if __name__ == "__main__":
    main()
