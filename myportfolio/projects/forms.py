from flask_wtf import FlaskForm
from wtforms import SubmitField, FloatField, DateField, SelectMultipleField
from wtforms.validators import DataRequired
import os
import pandas as pd

class iris_form(FlaskForm):
    sep_len = FloatField("Sepal Length",validators=[DataRequired()])
    sep_wid = FloatField("Sepal Width",validators=[DataRequired()])
    pet_len = FloatField("Petal Length",validators=[DataRequired()])
    pet_wid = FloatField("Petal Width",validators=[DataRequired()])
    submit = SubmitField("Predict")

class dash_form(FlaskForm):
    location = os.path.dirname(os.path.realpath(__file__))
    file_path = location+'/Data/companies.csv'
    stocks = pd.read_csv(file_path)
    options = list(zip(stocks['Symbol'],stocks['Name']))
    
    tickers = SelectMultipleField("Enter Stock Symbol",choices=options,default=['AAPL'])
    start_date = DateField("Start Date")
    end_date = DateField("End Date")
    submit = SubmitField("Submit")
    
