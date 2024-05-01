import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_data_names():
    return [name.removesuffix('.csv') for name in os.listdir('train') if '_preprocessed' not in name]


def read_data(name):
    train = pd.read_csv(os.path.join('train', f'{name}.csv'))
    test = pd.read_csv(os.path.join('test', f'{name}.csv'))
    return train, test


def scale_data(train, test):
    scaler = MinMaxScaler()
    scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return pd.DataFrame(train_scaled, columns=train.columns), pd.DataFrame(test_scaled, columns=test.columns)


def write_preprocessed_data(name, train_preprocessed, test_preprocessed):
    train_preprocessed.to_csv(os.path.join('train', f'{name}_preprocessed.csv'), index=False)
    test_preprocessed.to_csv(os.path.join('test', f'{name}_preprocessed.csv'), index=False)


def pipeline(name):
    write_preprocessed_data(name, *scale_data(*read_data(name)))


def main():
    data_names = get_data_names()

    for name in data_names:
        pipeline(name)


if __name__ == '__main__':
    main()
