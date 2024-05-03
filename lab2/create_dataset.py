import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer


class QuantileReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.05):
        self.threshold = threshold
        self.quantiles = {}

    def fit(self, X, y=None):
        for col in X.select_dtypes(include="number"):
            low_quantile = X[col].quantile(self.threshold)
            high_quantile = X[col].quantile(1 - self.threshold)
            self.quantiles[col] = (low_quantile, high_quantile)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in X.select_dtypes(include="number"):
            low_quantile, high_quantile = self.quantiles[col]
            rare_mask = ((X[col] < low_quantile) | (X[col] > high_quantile))
            if rare_mask.any():
                rare_values = X_copy.loc[rare_mask, col]
                replace_value = np.mean([low_quantile, high_quantile])
                if rare_values.mean() > replace_value:
                    X_copy.loc[rare_mask, col] = high_quantile
                else:
                    X_copy.loc[rare_mask, col] = low_quantile
        return X_copy


num_date = ["date"]
num_period = ["period"]
cat_day = ["day"]
num_nswprice = ["nswprice"]
num_nswdemand = ["nswdemand"]
num_vicprice = ["vicprice"]
num_vicdemand = ["vicdemand"]
num_transfer = ["transfer"]
cat_class = ["class"]


def get_preprocessing_transformer():
    num_pipe_date = "passthrough"
    num_pipe_period = "passthrough"
    cat_pipe_day = Pipeline([
        ("OneHotEncode", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
    ])
    num_pipe_nswprice = Pipeline([
        ("PowerTransform", PowerTransformer())
    ])
    num_pipe_nswdemand = Pipeline([
        ("QuantReplace", QuantileReplacer(threshold=0.001))
    ])
    num_pipe_vicprice = Pipeline([
        ("PowerTransform", PowerTransformer())
    ])
    num_pipe_vicdemand = Pipeline([
        ("QuantReplace", QuantileReplacer(threshold=0.001))
    ])
    num_pipe_transfer = Pipeline([
        ("QuantReplace", QuantileReplacer(threshold=0.001))
    ])
    cat_pipe_class = Pipeline([
        ("OneHotEncode", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer(transformers=[
        ("num_date", num_pipe_date, num_date),
        ("num_period", num_pipe_period, num_period),
        ("cat_day", cat_pipe_day, cat_day),
        ("num_nswprice", num_pipe_nswprice, num_nswprice),
        ("num_nswdemand", num_pipe_nswdemand, num_nswdemand),
        ("num_vicprice", num_pipe_vicprice, num_vicprice),
        ("num_vicdemand", num_pipe_vicdemand, num_vicdemand),
        ("num_transfer", num_pipe_transfer, num_transfer),
        ("cat_class", cat_pipe_class, cat_class)
    ])


def get_columns(transformer):
    return np.hstack([
        num_date,
        num_period,
        transformer.transformers_[2][1]["OneHotEncode"].get_feature_names_out(cat_day),
        num_nswprice,
        num_nswdemand,
        num_vicprice,
        num_vicdemand,
        num_transfer,
        cat_class
    ])


def get_raw_data():
    X, y = fetch_openml(data_id=151, return_X_y=True, as_frame=True, parser='auto')
    data = X
    data['class'] = y

    return data


def preprocess_data(data):
    transformer = get_preprocessing_transformer()
    transformed_data = transformer.fit_transform(data)
    preprocessed_data = pd.DataFrame(transformed_data, columns=get_columns(transformer))
    return preprocessed_data


def split_data(data, test_size=0.3):
    train_data, test_data = train_test_split(data, test_size=test_size)
    return train_data, test_data


def write_data(train_data, test_data):
    train_data.to_csv("train_data.csv", index=False)
    test_data.to_csv("test_data.csv", index=False)


def pipeline():
    write_data(*split_data(preprocess_data(get_raw_data())))


if __name__ == '__main__':
    pipeline()
