# BYTEUPRISE_ML_02
## Movie Ratings Prediction
This project demonstrates a machine learning pipeline for predicting movie ratings using various features extracted from a dataset. The pipeline includes data preprocessing, feature extraction, model training, and evaluation.

## Requirements
Python 3.8+
NumPy
Pandas
Matplotlib
scikit-learn
XGBoost

## Data Preprocessing
Drop unnecessary columns: Remove columns like overview and user_id.
Handle missing values: Use SimpleImputer to fill missing values with the most frequent values.
Label encoding: Convert categorical columns (genres, production_companies, production_countries, release_year) to numeric values using LabelEncoder.

## Feature Extraction
Use TfidfVectorizer to convert textual data into numerical features.

## Model Training
Use the XGBRegressor from XGBoost to train a model on the training data.

## Model Evaluation
Evaluate the model using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared score.
Plot the residuals to visualize the prediction errors.

## Usage
Prepare the dataset: Ensure the dataset is in the correct format and contains the required columns.
Run the preprocessing and training script: Execute the script to preprocess the data, extract features, train the model, and evaluate its performance.
Visualize results: Review the printed metrics and the residuals plot to understand the model's performance.

## Results
The model achieved an MSE of 0.98 and an R-squared score of 0.08.
The residuals plot helps identify patterns in the prediction errors.

