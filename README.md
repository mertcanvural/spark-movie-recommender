# Movie Recommendation System

A distributed movie recommendation system using Spark ALS collaborative filtering.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Download dataset (options: 100k, 1m, 10m, 25m)
python scripts/download_data.py 1m

# Process data
python scripts/common/preprocess_data.py

# Train Spark model
python scripts/train_spark_model.py

# Train NumPy model (single-machine)
python scripts/train_numpy_model.py

# Get recommendations for a user
python scripts/get_recommendation.py 1
```

The scripts handle:

- Dataset download and preparation
- Training ALS models (distributed and single-machine versions)
- Generating personalized movie recommendations
