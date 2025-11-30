import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import pickle


@st.cache_resource
def train_price_model(df):
    try:
        df_model = df.copy()

        feature_columns = [
            'brand', 'device_type', 'cpu_brand', 'cpu_cores', 'ram_gb',
            'storage_gb', 'gpu_brand', 'display_size_in'
        ]

        df_model = df_model[feature_columns + ['price']].dropna()

        label_encoders = {}
        for col in ['brand', 'device_type', 'cpu_brand', 'gpu_brand']:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le

        X = df_model[feature_columns]
        y = df_model['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        model_data = {
            'model': model,
            'label_encoders': label_encoders,
            'feature_columns': feature_columns,
            'metrics': {'mae': mae, 'r2': r2},
            'feature_importances': dict(zip(feature_columns, model.feature_importances_))
        }

        with open('price_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        return model_data

    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None


@st.cache_resource
def load_price_model(df):
    try:
        with open('price_model.pkl', 'rb') as f:
            model_data = pickle.load(f)

        st.success("Модель загружена из файла!")
        return model_data

    except FileNotFoundError:
        st.info("Обучаем новую модель...")
        return train_price_model(df)
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return None


if __name__ == "__main__":
    df = pd.read_csv("1/computer_prices_all.csv")
    train_price_model(df)
