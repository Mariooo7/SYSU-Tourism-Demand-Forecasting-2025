import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data_dict = {}
    files = glob.glob(file_path)
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        df = pd.read_excel(file, parse_dates=['Date'], date_format='%d-%m-%Y')
        data_dict[file_name] = df
    return data_dict

def split_data(df):
    df = df.copy()
    df.insert(0, 'unique_id', 1)
    split_idx = int(len(df) * 0.8)
    train = df[:split_idx]
    test = df[split_idx:]
    return train, test

def split_processed_data(df):
    df = df.copy()
    X = df.drop(columns=['Date', 'Total'])
    y = df['Total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
