from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import os
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
cors = CORS(app)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'templates', 'LinearRegressionModel.pkl')
DATA_PATH = os.path.join(ROOT_DIR, 'templates', 'quikr_car.csv')

model = None
car = None
load_error = None

missing_files = [path for path in (MODEL_PATH, DATA_PATH) if not os.path.exists(path)]
if missing_files:
    load_error = 'Missing file(s): ' + ', '.join(missing_files)
else:
    try:
        import sklearn.compose._column_transformer as _ct

        class _RemainderColsList(list):
            pass

        _ct._RemainderColsList = _RemainderColsList
    except Exception:
        pass

    try:
        model = pickle.load(open(MODEL_PATH, 'rb'))
    except Exception as ex:
        load_error = 'Failed to load model: ' + str(ex)

    try:
        car = pd.read_csv(DATA_PATH)
    except Exception as ex:
        load_error = 'Failed to load data: ' + str(ex)

@app.route('/', methods=['GET', 'POST'])
def index():
    if car is None:
        return render_template(
            'index.html',
            error_message=load_error,
            companies=[],
            years=[],
            fuel_types=[],
            company_models={},
        )

    companies = sorted(car['company'].dropna().unique())
    company_models = {
        company: sorted(car.loc[car['company'] == company, 'name'].dropna().unique())
        for company in companies
    }
    companies.insert(0, 'Select Company')
    
    # Convert year to int safely, drop any non-numeric values
    year_list = []
    for y in car['year'].dropna().unique():
        try:
            year_list.append(int(y))
        except (ValueError, TypeError):
            pass
    year = sorted(set(year_list), reverse=True)
    
    fuel_type = sorted(car['fuel_type'].dropna().unique())

    return render_template(
        'index.html',
        companies=companies,
        years=year,
        fuel_types=fuel_type,
        company_models=company_models,
    )


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    if model is None:
        return load_error or 'Prediction model is not available', 500

    if not company or company == 'Select Company' or not car_model:
        return 'Please select a valid company and model', 400

    try:
        year_value = int(year)
        kms_driven = int(float(driven))
    except (TypeError, ValueError):
        return 'Please enter a valid year and kilometres driven', 400

    input_df = pd.DataFrame([
        {
            'name': car_model,
            'company': company,
            'year': year_value,
            'kms_driven': kms_driven,
            'fuel_type': fuel_type,
        }
    ])

    try:
        prediction = model.predict(input_df)
    except Exception as ex:
        return f'Prediction failed: {ex}', 500

    return str(np.round(prediction[0], 2))


if __name__ == '__main__':
    app.run(debug=True)