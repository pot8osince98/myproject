# myportfolio/projects/views.py

from flask import render_template,url_for,request,redirect,Blueprint,flash,session
import yfinance as yf
import pandas as pd

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

from myportfolio.projects.forms import dash_form
from myportfolio.projects.models import (create_plot,create_hist,create_bar,get_ratios,
                                         sim_monte_carlo,optimal_weights)

@projects.route('/stocks_dash',methods=['GET','POST'])
def stocks_dash():
    form = dash_form()
    tickers = form.tickers.data
    start = form.start_date.data
    start = pd.Timestamp(start)
    
    if not form.validate_on_submit():
        form.end_date.data = form.end_date.default
    
    end = form.end_date.data
    end = pd.Timestamp(end)
    
    stocks = {}
    
    for tic in tickers:
        df = yf.download(tic)
        stocks[tic] = df
    
    plot = create_plot(stocks,start,end)
    hist = create_hist(stocks,start,end)
    bar = create_bar(stocks,start,end)
    ratios = get_ratios(stocks,start,end)
    mc_sim = sim_monte_carlo(stocks)
    opt_weights = optimal_weights()
        
    return render_template('stocks_dash.html',plot=plot,form=form,hist=hist,bar=bar,ratios=ratios,
                           mc_sim=mc_sim,opt_weights=opt_weights)

@projects.route('/tableau_dashboard',methods=['GET','POST'])
def tableau_dashboard():
    return render_template('tableau_dashboard.html')