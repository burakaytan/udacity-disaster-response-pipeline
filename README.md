# Disaster Response Pipeline Project

## Information
This repository was written in HTML and Python , and requires the following Python packages: 
 pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys.

## Project Overview
This code is designed to iniate a  web app which an emergency operators could exploit during a disaster, to classify a disaster text messages into several categories which then can be transmited to the responsible entity

## Files

- ETL Pipeline Preparation.ipynb: Description for workspace/data/process_data.py
- ML Pipeline Preparation.ipynb: Description for workspace/model/train_classifier.py
- workspace/data/process_data.py: A data cleaning pipeline that:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database
- workspace/model/train_classifier.py: A machine learning pipeline that:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements
This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025)