# myportfolio/projects/views.py

from flask import render_template,url_for,request,redirect,Blueprint,flash,session

projects = Blueprint('projects',__name__)

@projects.route('/')
def index():
    return render_template('projects.html')

@projects.route('/credit_card_fraud_detection')
def credit_card_fraud_detection():
    return render_template('credit_card_fraud_detection.html')

from myportfolio.projects.forms import iris_form
from myportfolio.projects.models import iris_predict

@projects.route('/iris',methods=['GET','POST'])
def iris():
    form = iris_form()
    if form.validate_on_submit():
        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data
        
        flower = {}
        flower['sep_len'] = float(session['sep_len'])
        flower['sep_wid'] = float(session['sep_wid'])
        flower['pet_len'] = float(session['pet_len'])
        flower['pet_wid'] = float(session['pet_wid'])
        
        iris_class = iris_predict(flower)
        
        flash(iris_class)
        return redirect(url_for('projects.iris'))
    return render_template('iris.html',form=form)

@projects.route('/home_price_predictor')
def home_price_predictor():
    return render_template('home_price_predictor.html')

@projects.route('/stocks_dash')
def stocks_dash():
    return render_template('stocks_dash.html')