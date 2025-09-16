import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("farming/dataset.csv")
df['crop'] = df['crop'].astype('category')
df['crop_code'] = df['crop'].cat.codes

X = df[['temperature', 'humidity', 'soil_moisture']]
y = df['crop_code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'farming/ml_model.pkl')
joblib.dump(df[['crop', 'crop_code']].drop_duplicates().set_index('crop_code').to_dict()['crop'], 'farming/crop_mapping.pkl')
