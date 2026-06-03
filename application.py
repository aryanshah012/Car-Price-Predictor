
from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')

@app.route('/')
def index():

    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    company_models = {}

    for company in companies:
        models = sorted(
            car[car['company'] == company]['name'].unique()
        )
        company_models[company] = models

    return render_template(
        'AryCarPP.html',
        companies=companies,
        company_models=company_models,
        years=years,
        fuel_types=fuel_types
    )

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():

    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    driven = int(request.form.get('kilo_driven'))

    test_df = pd.DataFrame({
        'name': [car_model],
        'company': [company],
        'year': [year],
        'kms_driven': [driven],
        'fuel_type': [fuel_type]
    })

    prediction = model.predict(test_df)

    return str(round(prediction[0], 2))


if __name__=='__main__':
    app.run(debug=True, port=5001)