from sklearn.datasets import fetch_openml


data = fetch_openml('Titanic', version=1, return_X_y=True, as_frame=True, parser='auto')

data_df = data[0].copy()
data_df['target'] = data[1]

data_df.to_csv('Titanic.csv', index=False)
