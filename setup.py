from setuptools import setup, find_packages

setup(
    name="movie-recommender",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pyspark==3.2.0",
        "numpy",
        "pandas",
        "flask",
        "flask-cors",
        "requests",
        "pytest",
    ],
    extras_require={
        "dev": [
            "jupyter",
            "matplotlib",
            "seaborn",
        ]
    },
)
