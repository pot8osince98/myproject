import joblib,os,json,plotly
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

def create_plot(df):
    
    fig = {'data':[{'x':df.index,'y':df['Adj Close']}]}

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def create_base_plot():
    df = pd.DataFrame({
      'Fruit': ['Apples', 'Oranges', 'Bananas', 'Apples', 'Oranges', 
      'Bananas'],
      'Amount': [4, 1, 2, 2, 4, 5],
      'City': ['SF', 'SF', 'SF', 'Montreal', 'Montreal', 'Montreal']
    })
    fig = px.bar(df, x='Fruit', y='Amount', color='City', 
      barmode='group')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON