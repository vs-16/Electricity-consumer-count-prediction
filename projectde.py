import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from requests.auth import HTTPBasicAuth
import json
import csv
from dotenv import load_dotenv 
import os
import streamlit as st 

load_dotenv()

st.title("Electricity Consumer count Prediction")

api_key=os.getenv("api_key")
headers={"authorization":api_key}
params = {
    "frequency": "annual",
    "data[0]": "capacity",
    "data[1]": "customers",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "offset": 0,
    "length": 5000,
    "api_key": api_key
}

r = requests.get("https://api.eia.gov/v2/electricity/state-electricity-profiles/net-metering/data/", params=params)

data=json.loads(r.content)
ac_dt=data["response"]["data"]

dt=pd.DataFrame(ac_dt)
dt=dt.drop(["state","capacity-units","customers-units","sector"],axis=1)
dt=dt.fillna(0)

label_encoder= LabelEncoder()
dt["stateName"]=label_encoder.fit_transform(dt["stateName"])
dt["technology"]=label_encoder.fit_transform(dt["technology"])
dt["sectorName"]=label_encoder.fit_transform(dt["sectorName"])

x=dt[["stateName","technology","sectorName","capacity","period"]]
y=dt["customers"]
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

model= LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse= mean_squared_error(y_test,y_pred)
r2= r2_score(y_test,y_pred)

st_name=st.text_input(label="US State name")
tech_name=st.text_input(label="Technology used for electricity production")
sec_name=st.text_input(label="Sector consuming electricity")
cap=st.number_input(label="Capacity of Electricity supplied (in Megawatts)")
year=st.date_input(label="Enter the year")
year=year.year
btn=st.button(label="Submit")

new_input = pd.DataFrame({
    'stateName':[st_name] ,
    'technology': [tech_name],
    'sectorName': [sec_name],
   'capacity': [cap],
    'period': [year]})

new_input["stateName"]=label_encoder.fit_transform(new_input["stateName"])
new_input["technology"]=label_encoder.fit_transform(new_input["technology"])
new_input["sectorName"]=label_encoder.fit_transform(new_input["sectorName"])
pred_value= model.predict(new_input)

if btn and st_name!='' and tech_name!='' and sec_name!='': 
    st.write("Predicted consumer count-")
    pred=np.floor(pred_value)[0]
    st.write(f'## {pred}') 
    fig,ax=plt.subplots()
    ax.scatter(y_test,y_pred)
    ax.set_xlabel("Actual num of consumers")
    ax.set_ylabel("Predicted num of consumers")
    ax.set_title("Actual vs Predicted")
    regression_line= np.polyfit(y_test,y_pred,1)
    ax.plot(y_test,np.polyval(regression_line,y_test),color="Green",label="Reg line")
    ax.legend()
    st.pyplot(fig)
