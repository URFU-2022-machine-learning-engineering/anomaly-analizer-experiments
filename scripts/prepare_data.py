import numpy as np
import pandas as pd


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)

    # Data cleaning and preparation
    df.dropna(subset=["startTime"], inplace=True)
    df["startTime"] = pd.to_datetime(df["startTime"])
    df.set_index("startTime", inplace=True)
    df.sort_index(inplace=True)
    df = df["2024-04-02":]

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
    upload_spans['upload_duration_per_file_size'] = upload_spans['duration'] / (upload_spans['matched_file_size'] + epsilon)

    df.loc[upload_spans.index, 'upload_duration_per_file_size'] = upload_spans['upload_duration_per_file_size']
    df['upload_duration_per_file_size'] = df['upload_duration_per_file_size'].fillna(0)

    # Dropping unnecessary columns
    df.dropna(axis=1, how='all', inplace=True)
    columns_to_drop = [
        "tag_net.sock.peer.port", "tag_internal.span.format", "http.status_code", "file.name",
        "minio.file.size", "file_type", "whisperTranscribeURL", "time", "uuid", "minio.bucket",
        "tag_minio.bucket", "tag_error.message", "tag_uuid", "tag_http.scheme", "tag_net.host.name",
        "tag_net.protocol.version", "tag_http.route", "tag_http.target", "file_name", "file",
        "tag_http.url", "bucket"
    ]
    df.drop(columns=columns_to_drop, inplace=True, axis=1)

    # Filling missing data
    columns_to_fill = ['tag_http.client_ip', 'tag_user_agent.original', 'tag_net.sock.peer.addr', 'tag_file.type']
    for column in columns_to_fill:
        df[column] = df.groupby('traceID')[column].transform(lambda x: x.ffill().bfill())

    return df


def main():
    df = load_and_prepare_data("data/data_with_anomalies.parquet")
    df.to_csv("data/prepared_data.csv", index=False)
    print(df.sample(10))  # Display a sample of the prepared data for verification


if __name__ == "__main__":
    main()
