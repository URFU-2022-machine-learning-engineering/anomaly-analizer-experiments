from pathlib import Path
import pandas as pd
import pickle
import json
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


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
    metrics_dir = Path(__file__).parents[1] / 'metrics'

    df = pd.read_csv(data_path, low_memory=False)
    df = apply_encoders(df, label_encoders)
    df = df.drop(columns=['spanID', 'traceID', 'tag_http.client_ip', 'tag_otel.status_description', 'tag_user_agent.original'])

    X = df.drop(columns=['anomaly'])  # Exclude the target variable for predictions
    y = df['anomaly']

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)

    # Save classification report and accuracy
    metrics_dir = Path('metrics')
    with open(metrics_dir / 'accuracy.json', 'w') as f:
        json.dump({"accuracy": accuracy}, f)
    with open(metrics_dir / 'classification_report.json', 'w') as f:
        json.dump(report, f)
    labels = ['Non-anomalous', 'Anomalous']

    # Creating a DataFrame from the confusion matrix
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
    conf_matrix_df.to_csv(f"{metrics_dir}/confusion_matrix.csv")

    # Save feature importances
    if isinstance(model, RandomForestClassifier):
        feature_importances = model.feature_importances_
        features = X.columns
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        })
        feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
        feature_importance_df.to_csv(metrics_dir / 'feature_importances.csv', index=False)


def main():
    model_path = Path(__file__).parents[1] / 'model' / 'rf_model.pkl'
    encoders_path = Path(__file__).parents[1] / 'model' / 'encoders.pkl'
    data_path = Path(__file__).parents[1] / 'data' / 'prepared_data.csv'
    model, label_encoders = load_model(model_path=model_path, encoders_path=encoders_path)
    evaluate_model(model=model, label_encoders=label_encoders, data_path=data_path)


if __name__ == "__main__":
    main()
