import joblib,os,json,plotly
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

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

def create_plot(stocks):
    
    traces =[]
    
    for tic, df in stocks.items():
        traces.append(go.Scatter(x=df.index,y=df['Adj Close'],mode='lines',name=tic))
    
    layout = go.Layout(template='ggplot2')
    
    fig = go.Figure(data=traces,layout=layout)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_hist(stocks):
    
    fig = go.Figure()
    
    for tic, df in stocks.items():
        fig.add_trace(go.Histogram(x=df['Daily Returns'],name=tic,
                                   xbins=dict(start=-0.2,end=0.2,size=0.005)))
    
    fig.update_layout(title='Daily Returns',barmode='overlay',template='ggplot2')
    fig.update_traces(opacity=0.3)
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_bar(stocks):
    
    data = []
    
    for tic, df in stocks.items():
        pct_change = round(100*(df.loc['2023-01-03']['Adj Close']-df.loc['2021-01-04']['Adj Close'])/df.loc['2021-01-04']['Adj Close'],2)
        data.append([tic,pct_change])
    
    df = pd.DataFrame(data=data,columns=['Symbol','Pct Change'])
    
    df['Color'] = np.where(df['Pct Change']<0,'red','green')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=df['Pct Change'],y=df['Symbol'],marker_color=df['Color'],orientation='h',
                         text=df['Pct Change'],textposition='auto'))
    
    fig.update_layout(barmode='stack',template='ggplot2')
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON