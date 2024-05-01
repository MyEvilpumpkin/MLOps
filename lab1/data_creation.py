import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_data(a=1, b=0, d_amp=2):
    x = np.linspace(0, 100, 101)
    d = np.random.random(len(x))*d_amp*2 - d_amp
    y = a * x + b + d

    return pd.DataFrame({'feature': x, 'target': y})


def split_data(data, test_size=0.25):
    return train_test_split(data, test_size=test_size)


def write_data(name, train, test):
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)
    train.to_csv(os.path.join('train', f'{name}.csv'), index=False)
    test.to_csv(os.path.join('test', f'{name}.csv'), index=False)


def pipeline(name, creation_parameters):
    write_data(name, *split_data(create_data(**creation_parameters)))


def main():
    data_creation_parameters = {
        'data_0': {
            'a': 3,
            'b': 2
        },
        'data_1': {
            'a': -2,
            'b': 1
        },
        'data_2': {
            'a': 1,
            'b': -5
        }
    }

    for name in data_creation_parameters:
        pipeline(name, data_creation_parameters[name])


if __name__ == '__main__':
    main()
