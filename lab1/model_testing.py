import os

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import r2_score


def get_model_names():
    return [name.removesuffix('.joblib') for name in os.listdir('models')]


def read_preprocessed_test_data(name):
    test_preprocessed = pd.read_csv(os.path.join('test', f'{name}_preprocessed.csv'))
    return test_preprocessed


def read_model(name):
    model = joblib.load(os.path.join('models', f'{name}.joblib'))
    return model


def test_model(name, model, test):
    predicted = model.predict(test[['feature']])

    mse = mse_score(test['target'], predicted)
    r2 = r2_score(test['target'], predicted)

    print(f'Name: {name}, MSE: {mse}, R2: {r2}')


def pipeline(name):
    test_model(name, read_model(name), read_preprocessed_test_data(name))


def main():
    model_names = get_model_names()

    for name in model_names:
        pipeline(name)


if __name__ == '__main__':
    main()
