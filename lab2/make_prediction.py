import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.metrics import r2_score


target_column = 'nswprice'


def read_test_data():
    test_data = pd.read_csv('test_data.csv')
    return test_data


def read_model():
    model = joblib.load('model.joblib')
    return model


def test_model(model, data):
    X, y = data.drop(columns=target_column), data[target_column]

    predicted = model.predict(X)

    mse = mse_score(y, predicted)
    r2 = r2_score(y, predicted)

    print(f'MSE: {mse}, R2: {r2}')


def pipeline():
    test_model(read_model(), read_test_data())


if __name__ == '__main__':
    pipeline()
