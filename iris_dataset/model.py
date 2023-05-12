import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import load

# Load the trained model from disk
model_path = 'iris_dataset/random_forest.joblib'
rf_model = load(model_path)

def predict(input_data):
    # Convert the input data to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Use the trained model to make predictions on the input data
    prediction = rf_model.predict(input_df)

    # Define a dictionary that maps the numeric predictions to the iris type
    iris_types = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

    # Map the numeric prediction to the iris type
    result = iris_types[prediction[0]]

    # Return the prediction as a string
    return result
