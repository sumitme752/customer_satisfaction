import pandas as pd
import numpy as np
import pickle
import os

from sklearn import set_config
set_config(display='diagram')

from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("*"*10,'Reading the csv file',"*"*10)
df = pd.read_csv('./data/train.csv')

#Dropping the columns
df = df.drop(columns=['Unnamed: 0','id','Gender'], axis=1)


df['target'] = df['satisfaction'].apply(lambda x: 1 if x=='satisfied' else 0)

df = df.drop(columns=['satisfaction'], axis=1)

x = df.drop('target', axis=1)
y = df['target']

# Define numerical and categorical columns
num_features = x.select_dtypes(exclude='object').columns
cat_features = x.select_dtypes(include='object').columns


# Create a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler())
        ]), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ]
)

# Create a pipeline that combines preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred)
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)



# print(X_train.columns)


#Saving the model

# # Define the directory and file path
# model_dir = './model/'
# model_file = os.path.join(model_dir, 'my_model.pkl')

# # Check if the directory exists, if not, create it
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

# # Save or update the model
# with open(model_file, 'wb') as file:
#     pickle.dump(pipeline, file)

# print(f"Model saved or updated at {model_file}")
