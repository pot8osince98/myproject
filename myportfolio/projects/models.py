import joblib

def iris_predict(flower_example):
    
    sep_len = flower_example['sep_len']
    sep_wid = flower_example['sep_wid']
    pet_len = flower_example['pet_len']
    pet_wid = flower_example['pet_wid']
    
    flower = [[sep_len,sep_wid,pet_len,pet_wid]]
    
    scaler = joblib.load("iris_scaler.pkl")
    
    model = joblib.load("iris_model.pkl")
    
    flower = scaler.transform(flower)
    
    iris_class = model.predict(flower)[0].title()
    
    return iris_class
