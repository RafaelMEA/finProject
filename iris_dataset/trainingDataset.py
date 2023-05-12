# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# from joblib import dump

# Load the dataset
iris_df = pd.read_csv('iris.csv')

# Preprocessing: Convert species to numerical labels
species_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
iris_df['species'] = iris_df['species'].map(species_map)

# Split the data into training and testing sets
X = iris_df.drop('species', axis=1)
y = iris_df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [3, 5, None],
              'min_samples_leaf': [1, 2, 4]}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

# Cross-validation
rf_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Mean score:", np.mean(scores))

# Model selection
rf_model = RandomForestClassifier(**grid_search.best_params_, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", accuracy)

# Save the trained model to disk
#model_path = 'random_forest.joblib'
#dump(rf_model, model_path)