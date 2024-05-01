import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression


def get_preprocessed_data_names():
    return [name.removesuffix('_preprocessed.csv') for name in os.listdir('train') if '_preprocessed' in name]


def read_preprocessed_train_data(name):
    train_preprocessed = pd.read_csv(os.path.join('train', f'{name}_preprocessed.csv'))
    return train_preprocessed


def train_model(train):
    model = LinearRegression()
    model.fit(train[['feature']], train['target'])
    return model


def write_model(name, model):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, os.path.join('models', f'{name}.joblib'))


def pipeline(name):
    write_model(name, train_model(read_preprocessed_train_data(name)))


def main():
    data_names = get_preprocessed_data_names()

    for name in data_names:
        pipeline(name)


if __name__ == '__main__':
    main()
