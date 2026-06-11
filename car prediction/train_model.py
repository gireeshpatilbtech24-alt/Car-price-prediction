import os
import re
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'templates', 'quikr_car.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'templates', 'LinearRegressionModel.pkl')


def clean_price(x):
    if isinstance(x, str):
        x = x.replace(',', '').strip()
        if x.lower().startswith('ask'):
            return None
    try:
        return float(x)
    except Exception:
        return None


def clean_kms(x):
    if isinstance(x, str):
        x = x.replace(',', '').lower()
        x = re.sub(r'[^0-9.]', '', x)
    try:
        return float(x)
    except Exception:
        return None


def main():
    df = pd.read_csv(DATA_PATH)
    df['Price'] = df['Price'].apply(clean_price)
    df['kms_driven'] = df['kms_driven'].apply(clean_kms)
    df = df.dropna(subset=['Price', 'kms_driven', 'name', 'company', 'year', 'fuel_type'])
    df['year'] = df['year'].astype(int)

    X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = df['Price'].astype(float)

    cat_cols = ['name', 'company', 'fuel_type']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', LinearRegression()),
    ])

    pipeline.fit(X, y)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)

    print('Trained and saved model to', MODEL_PATH)


if __name__ == '__main__':
    main()
