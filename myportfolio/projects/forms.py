from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,FloatField
from wtforms.validators import DataRequired

class iris_form(FlaskForm):
    sep_len = FloatField("Sepal Length",validators=[DataRequired()])
    sep_wid = FloatField("Sepal Width",validators=[DataRequired()])
    pet_len = FloatField("Petal Length",validators=[DataRequired()])
    pet_wid = FloatField("Petal Width",validators=[DataRequired()])
    submit = SubmitField("Predict")

