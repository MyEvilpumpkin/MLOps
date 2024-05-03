import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


target_column = 'nswprice'


def read_train_data():
    train_data = pd.read_csv('train_data.csv')
    return train_data


def train_model(data):
    X, y = data.drop(columns=target_column), data[target_column]
    model = GradientBoostingRegressor(n_estimators=150, subsample=0.7)
    model.fit(X, y)
    return model


def write_model(model):
    joblib.dump(model, 'model.joblib')


def pipeline():
    write_model(train_model(read_train_data()))


if __name__ == '__main__':
    pipeline()
