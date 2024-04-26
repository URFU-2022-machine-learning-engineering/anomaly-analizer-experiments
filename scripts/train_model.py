from pathlib import Path

import pandas as pd
import pickle
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


def encode_features(df):
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    label_encoders = {}
    for column in non_numeric_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    return df, label_encoders


def train_model(data_path, model_path, encoders_path, params):
    df = pd.read_csv(data_path, low_memory=False)
    df, label_encoders = encode_features(df)
    X = df.drop(columns=['anomaly', 'spanID', 'traceID', 'tag_http.client_ip', 'tag_otel.status_description',
                         'tag_user_agent.original'])
    y = df['anomaly']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    with open(encoders_path, 'wb') as file:
        pickle.dump(label_encoders, file)


def main():
    data_path = Path(__file__).parents[1] / 'data' / 'prepared_data.csv'
    model_path = Path(__file__).parents[1] / 'model' / 'rf_model.pkl'
    encoders_path = Path(__file__).parents[1] / 'model' / 'encoders.pkl'
    params_path = Path(__file__).parents[1] / "params" / "params.yaml"
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    train_model(data_path=data_path, model_path=model_path, encoders_path=encoders_path, params=params['model_params'])


if __name__ == "__main__":
    main()
