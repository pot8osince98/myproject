from flask_wtf import FlaskForm
from wtforms.fields import SubmitField, FloatField, DateField, SelectMultipleField
from wtforms.validators import DataRequired, ValidationError
from datetime import datetime
from dateutil.relativedelta import relativedelta
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
    
    tickers = SelectMultipleField("Enter Stock Symbol",choices=options,
                                  default=['GOOGL','AMZN','AAPL','MSFT'],
                                  validators=[DataRequired()])
    start_date = DateField("Start Date",default=datetime.today().date() - relativedelta(years=5),
                           validators=[DataRequired()])
    end_date = DateField("End Date",default=datetime.today().date(),
                         validators=[DataRequired()])
    submit = SubmitField("Submit")
    
    def validate_end_date(form, field):
        if field.data < form.start_date.data:
            raise ValidationError("End date must not be earlier than start date.")