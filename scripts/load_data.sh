#!/bin/bash

bash "$HOME/data_load.sh"
cp "$HOME/urfu/anomalies-detection/enriched_spans.csv" "data/enriched_spans.csv" || exit 1