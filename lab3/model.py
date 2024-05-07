import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def create():
    # Data loading
    dataset = load_iris()

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

    # Model creation
    model = Pipeline([('scaler', StandardScaler()), ('classifier', LogisticRegression())])

    # Model training
    model.fit(X_train, y_train)

    # Model testing
    print(classification_report(y_test, model.predict(X_test)))

    # Model saving
    model.feature_names = dataset.feature_names
    model.target_names = dataset.target_names
    joblib.dump(model, 'model.joblib')


def load():
    # Model loading
    return joblib.load('model.joblib')
