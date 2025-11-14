"""
Data processing module for telemetry data
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json

def clean_data(telemetry_data):
    """
    Clean and normalize telemetry data
    """
    if not isinstance(telemetry_data, dict):
        raise ValueError("Telemetry data must be a dictionary")

    # Ensure required fields
    required_fields = ['ver', 'name', 'time']
    for field in required_fields:
        if field not in telemetry_data:
            telemetry_data[field] = None

    # Parse timestamp
    if telemetry_data['time']:
        try:
            telemetry_data['parsed_time'] = datetime.fromisoformat(telemetry_data['time'].replace('Z', '+00:00'))
        except:
            telemetry_data['parsed_time'] = None

    return telemetry_data

def extract_features(data_list):
    """
    Extract features from a list of telemetry data
    """
    df = pd.DataFrame(data_list)

    # Time-based features
    if 'parsed_time' in df.columns:
        df['hour'] = df['parsed_time'].dt.hour
        df['day_of_week'] = df['parsed_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # Operation-based features
    if 'data' in df.columns:
        df['op_length'] = df['data'].apply(lambda x: len(str(x)) if x else 0)
        df['has_pfn'] = df['data'].apply(lambda x: 1 if isinstance(x, dict) and 'PFN' in x else 0)

    # Fill missing values
    df = df.fillna(0)

    return df

def prepare_for_ml(data_list):
    """
    Prepare data for machine learning models
    """
    cleaned_data = [clean_data(item) for item in data_list]
    features_df = extract_features(cleaned_data)

    # Select relevant features for ML
    feature_columns = ['hour', 'day_of_week', 'is_weekend', 'op_length', 'has_pfn']
    X = features_df[feature_columns].values

    return X, features_df
