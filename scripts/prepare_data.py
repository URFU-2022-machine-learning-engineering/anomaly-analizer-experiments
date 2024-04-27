import pandas as pd
import numpy as np
from pathlib import Path

import yaml


def mark_anomalies(df_path: Path, params: dict, save_path: Path):
    df = pd.read_csv(df_path, low_memory=False)
    df.dropna(subset=["startTime"], inplace=True)
    df["startTime"] = pd.to_datetime(df["startTime"])
    df.set_index("startTime", inplace=True)
    df.sort_index(inplace=True)
    df = df["2024-04-02":]

    # Convert relevant columns to correct data types for analysis
    df['tag_http.status_code'] = pd.to_numeric(df['tag_http.status_code'], errors='coerce')
    df['tag_error'] = df['tag_error'].apply(lambda x: x == 'True' if isinstance(x, str) else x)

    # Define anomaly conditions
    condition_http_status = df['tag_http.status_code'] >= params["status_code"]
    condition_duration = df['duration'] >= params["duration"]  # Assuming duration is in microseconds
    condition_specific_user = (df['tag_http.client_ip'] == params["ip_address"]) & \
                              (df['tag_user_agent.original'] == params["ua"])
    condition_status_or_error = (df['tag_otel.status_code'] == 'ERROR') | (df['tag_error'] == True)

    # Combine conditions
    anomalies = condition_http_status | condition_duration | condition_specific_user | condition_status_or_error

    # Mark trace IDs as anomalous
    anomalous_trace_ids = df.loc[anomalies, 'traceID'].unique()

    # Apply anomaly marking to the entire dataset
    df['anomaly'] = df['traceID'].isin(anomalous_trace_ids)

    # Save the dataset with anomalies marked
    df.to_csv(save_path, index=False)


def load_and_prepare_data(params: dict, df_anomalies_path: Path):
    # Load the dataset with anomalies marked
    df = pd.read_csv(df_anomalies_path, low_memory=False)

    epsilon = 1e-9
    df['span_duration_per_file_size'] = df['duration'] / (df['tag_minio.file.size'] + epsilon)

    # Filtering operations and feature computation
    upload_spans = df[df['operationName'] == '/upload']
    upload_to_minio_spans = df[df['operationName'] == 'UploadToMinio']
    upload_to_minio_indexed = upload_to_minio_spans.set_index('traceID')

    def get_file_size_from_upload_to_minio(row):
        try:
            return upload_to_minio_indexed.loc[row['traceID'], 'tag_minio.file.size']
        except KeyError:
            return np.nan

    upload_spans['matched_file_size'] = upload_spans.apply(get_file_size_from_upload_to_minio, axis=1)
    upload_spans['upload_duration_per_file_size'] = upload_spans['duration'] / (
            upload_spans['matched_file_size'] + epsilon)

    df.loc[upload_spans.index, 'upload_duration_per_file_size'] = upload_spans['upload_duration_per_file_size']
    df['upload_duration_per_file_size'] = df['upload_duration_per_file_size'].fillna(0)

    # Dropping unnecessary columns
    df.dropna(axis=1, how='all', inplace=True)
    columns_to_drop = params['columns_to_drop']
    df.drop(columns=columns_to_drop, inplace=True, axis=1)

    # Filling missing data
    columns_to_fill = ['tag_http.client_ip', 'tag_user_agent.original', 'tag_net.sock.peer.addr', 'tag_file.type']
    for column in columns_to_fill:
        df[column] = df.groupby('traceID')[column].transform(lambda x: x.ffill().bfill())

    return df


def main():
    params_path = Path(__file__).parents[1] / "params.yaml"
    prepared_df_path = Path(__file__).parents[1] / 'data' / 'prepared_data.csv'
    df_path = Path(__file__).parents[1] / "data" / "enriched_spans.csv"
    df_with_anomalies = Path(__file__).parents[1] / "data" / "data_with_anomalies.csv"

    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)

    mark_anomalies(df_path=df_path, params=params["prepare_params"], save_path=df_with_anomalies)  # First function call to add anomalies
    prepared_df = load_and_prepare_data(params=params["prepare_params"], df_anomalies_path=df_with_anomalies)  # Second function to load and further prepare data
    prepared_df.to_csv(prepared_df_path, index=False)


if __name__ == "__main__":
    main()
