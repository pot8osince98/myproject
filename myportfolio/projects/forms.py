from flask_wtf import FlaskForm
from wtforms import SubmitField, FloatField, DateField, SelectMultipleField
from wtforms.validators import DataRequired

class iris_form(FlaskForm):
    sep_len = FloatField("Sepal Length",validators=[DataRequired()])
    sep_wid = FloatField("Sepal Width",validators=[DataRequired()])
    pet_len = FloatField("Petal Length",validators=[DataRequired()])
    pet_wid = FloatField("Petal Width",validators=[DataRequired()])
    submit = SubmitField("Predict")

class dash_form(FlaskForm):
    stocks = SelectMultipleField("Enter Stock Symbol",choices=[('cpp', 'C++'), ('py', 'Python'), ('text', 'Plain Text')])
    start_date = DateField("Start Date")
    end_date = DateField("End Date")
    submit = SubmitField("Submit")
    
