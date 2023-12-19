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

def json_to_csv_conversion(data):
    
    electricity_data = data['response']['data']
 
    data_file = open('data_file.csv', 'w')
 
    csv_writer = csv.writer(data_file)
 
    count = 0
 
    for val in electricity_data:
        if count == 0:
 
            header = val.keys()
            csv_writer.writerow(header)
            count += 1
 
        csv_writer.writerow(val.values())
 
    data_file.close()
load_dotenv()
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
json_to_csv_conversion(r.json())
dt=pd.read_csv("data_file.csv")
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

new_input = pd.DataFrame({
    'stateName':["Arizona"] ,
    'technology': ["All Technologies"],
    'sectorName': ["commercial"],
   'capacity': [1611.85],
    'period': [2020]})
new_input["stateName"]=label_encoder.fit_transform(new_input["stateName"])
new_input["technology"]=label_encoder.fit_transform(new_input["technology"])
new_input["sectorName"]=label_encoder.fit_transform(new_input["sectorName"])
pred_value= model.predict(new_input)
print(np.floor(pred_value)[0])

plt.scatter(y_test,y_pred)
plt.xlabel("Actual num of customers")
plt.ylabel("predicted num of customers")
plt.title("Actual vs Predicted")
regression_line= np.polyfit(y_test,y_pred,1)
plt.plot(y_test,np.polyval(regression_line,y_test),color="Green",label="Reg line")
plt.legend()
plt.show()
