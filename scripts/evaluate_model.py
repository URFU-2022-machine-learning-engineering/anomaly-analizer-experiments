from pathlib import Path

import pandas as pd
import pickle
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_model(model_path, encoders_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)
    return model, label_encoders


def apply_encoders(df, label_encoders):
    for column, encoder in label_encoders.items():
        df[column] = encoder.transform(df[column].astype(str))
    return df


def evaluate_model(model, label_encoders, data_path):
    df = pd.read_csv(data_path, low_memory=False)
    df = apply_encoders(df, label_encoders)
    df = df.drop(
        columns=['spanID', 'traceID', 'tag_http.client_ip', 'tag_otel.status_description', 'tag_user_agent.original'])

    X = df.drop(columns=['anomaly'])  # Exclude the target variable for predictions
    y = df['anomaly']  # Actual labels

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    metrics_dir = Path(__file__).parents[1] / 'metrics'
    with open(f"{metrics_dir}/accuracy.json", 'w') as f:
        f.write(f'{{"accuracy": {accuracy}}}')
    with open(f"{metrics_dir}/classification_report.json", 'w') as f:
        f.write(report)

    conf_matrix = confusion_matrix(y, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"{metrics_dir}/conf_matrix.png")


def main():
    model_path = Path(__file__).parents[1] / 'model' / 'rf_model.pkl'
    encoders_path = Path(__file__).parents[1] / 'model' / 'encoders.pkl'
    data_path = Path(__file__).parents[1] / 'data' / 'prepared_data.csv'
    model, label_encoders = load_model(model_path=model_path, encoders_path=encoders_path)
    evaluate_model(model=model, label_encoders=label_encoders, data_path=data_path)


if __name__ == "__main__":
    main()
