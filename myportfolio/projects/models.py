import joblib,os,json,plotly
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objs as go
import sklearn
from scipy.stats import linregress
from scipy.optimize import minimize

def iris_predict(flower_example):
    
    sep_len = flower_example['sep_len']
    sep_wid = flower_example['sep_wid']
    pet_len = flower_example['pet_len']
    pet_wid = flower_example['pet_wid']
    
    flower = [[sep_len,sep_wid,pet_len,pet_wid]]
    
    location = os.path.dirname(os.path.realpath(__file__))
    
    scaler = joblib.load(os.path.join(location, 'iris_scaler.pkl'))
    
    model = joblib.load(os.path.join(location, 'iris_model.pkl'))
    
    flower = scaler.transform(flower)
    
    iris_class = model.predict(flower)[0].title()
    
    return iris_class

def create_plot(stocks,start_date,end_date):
    
    traces =[]
    
    for tic, df in stocks.items():
        
        start = start_date
        end = end_date
        
        if start_date not in df.index:
            start = min(date for date in df.index if date>start_date)
        if end_date not in df.index:
            end = max(date for date in df.index if date<end_date)
            
        traces.append(go.Scatter(x=df.loc[start:end].index,y=df.loc[start:end]['Adj Close'],mode='lines',name=tic))
    
    layout = go.Layout(template='ggplot2')
    
    fig = go.Figure(data=traces,layout=layout)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_hist(stocks,start_date,end_date):
    
    fig = go.Figure()
    
    for tic, df in stocks.items():
        
        start = start_date
        end = end_date
        
        if start_date not in df.index:
            start = min(date for date in df.index if date>start_date)
        if end_date not in df.index:
            end = max(date for date in df.index if date<end_date)
            
        df['Daily Returns'] = df.loc[start:end]['Adj Close'].pct_change()
        fig.add_trace(go.Histogram(x=df['Daily Returns'],name=tic,
                                   xbins=dict(start=-0.2,end=0.2,size=0.005)))
    
    fig.update_layout(title='Daily Returns',barmode='overlay',template='ggplot2')
    fig.update_traces(opacity=0.3)
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_bar(stocks,start_date,end_date):
    
    data = []
    
    for tic, df in stocks.items():
                
        start = start_date
        end = end_date
        
        if start_date not in df.index:
            start = min(date for date in df.index if date>start_date)
        if end_date not in df.index:
            end = max(date for date in df.index if date<end_date)
            
        pct_change = round(100*(df.loc[end]['Adj Close']-df.loc[start]['Adj Close'])/df.loc[start]['Adj Close'],2)
        data.append([tic,pct_change])
    
    df = pd.DataFrame(data=data,columns=['Symbol','Pct Change'])
    
    df['Color'] = np.where(df['Pct Change']<0,'red','green')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=df['Pct Change'],y=df['Symbol'],marker_color=df['Color'],orientation='h',
                         text=df['Pct Change'].apply(str)+'%',textposition='auto'))
    
    fig.update_layout(title="Percentage Change",barmode='stack',template='ggplot2')
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def get_ratios(stocks,start_date,end_date):
    
    ratios = {}
    
    sp500 = yf.download('SPY')
    
    sp500['Daily Returns'] = sp500['Adj Close'].pct_change()
    
    for tic, df in stocks.items():
        
        start = start_date
        end = end_date
    
        if start_date not in df.index:
            start = min(date for date in df.index if date>start_date)
        if end_date not in df.index:
            end = max(date for date in df.index if date<end_date)
            
        data = []
        
        mean = df.loc[start:end]['Daily Returns'].dropna().mean()
        std = df.loc[start:end]['Daily Returns'].dropna().std()
        
        sharpe_ratio = (mean/std)*(252**0.5)
        sharpe_ratio = '{:.2e}'.format(sharpe_ratio)
        
        data.append(sharpe_ratio)
        
        start = max(start,sp500.index[0])
        
        if (start > sp500.index[0]):
            beta,alpha,_,_,_ = linregress(sp500.loc[start:end].iloc[1:]['Daily Returns'],
                                          df['Daily Returns'].dropna())
        else:
            beta,alpha,_,_,_ = linregress(sp500['Daily Returns'].dropna(),
                                          df.loc[start:end].iloc[1:]['Daily Returns'])
        
        alpha = '{:.2e}'.format(alpha)
        beta = '{:.2e}'.format(beta)
        
        data.append(alpha)
        data.append(beta)
        
        ratios[tic] = data
        
    return ratios

log_returns = pd.DataFrame()

log_returns_cov = pd.DataFrame()

def calculate_returns(weights,log_rets):
    
    return np.sum(log_rets.mean()*weights) * 252

def calculate_volatility(weights,log_rets_cov):
    
    annualized_cov = np.dot(log_rets_cov*252,weights)
    vol = np.dot(weights.transpose(),annualized_cov)
    
    return vol**0.5

def sim_monte_carlo(stocks):
    global log_returns,log_returns_cov
    
    daily_rets = []
    
    for df in stocks.values():
        daily_rets.append(df['Daily Returns'])
    
    daily_returns = pd.concat(daily_rets,axis=1)
    daily_returns.columns = stocks.keys()
    
    log_returns = np.log(1+daily_returns.dropna())
    log_returns_cov = log_returns.cov()
    
    N = len(log_returns.columns)
    
    def gen_weights(N):
        weights = np.random.random(N)
        return weights/np.sum(weights)
    
    mc_portfolio_returns = []
    mc_portfolio_vol = []
    mc_weights = []

    for sim in range(10000):
        weights = gen_weights(N)
        mc_weights.append(weights)
        sim_returns = calculate_returns(weights,log_returns)
        mc_portfolio_returns.append(sim_returns)
        sim_vol = calculate_volatility(weights,log_returns_cov)
        mc_portfolio_vol.append(sim_vol)
        
    mc_sr = np.array(mc_portfolio_returns)/np.array(mc_portfolio_vol)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mc_portfolio_vol,y=mc_portfolio_returns,
                             mode='markers',marker=dict(color=mc_sr,
                                                        colorscale='Plasma',
                                                        showscale=True)))
    fig.update_layout(title='Monte Carlo Simulation (10,000 samples)',
                      template='ggplot2')
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def optimal_weights():
    
    N = len(log_returns.columns)
    equal_weights = np.array(N * [1/N])
    
    def func_to_minimize(weights):
        return -1 * (calculate_returns(weights,log_returns)/
                     calculate_volatility(weights,log_returns_cov))
    
    bounds = tuple((0,1) for n in range(N))
    
    sum_constraint = ({'type':'eq','fun':lambda weights: np.sum(weights)-1})
    
    opt_weight = minimize(fun=func_to_minimize,x0=equal_weights,
                          bounds=bounds,constraints=sum_constraint).x
    
    opt_weight = np.round(opt_weight*100,2)
    
    fig = go.Figure()
    
    for i in range(len(opt_weight)):
        fig.add_trace(go.Bar(x=[opt_weight[i]],name=log_returns.columns[i],
                             text=str(opt_weight[i])+'%',textposition='auto'))
    
    fig.update_layout(title='Optimal Weights',barmode='stack',template='ggplot2',
                      yaxis={'visible': False, 'showticklabels': False})
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON