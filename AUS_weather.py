import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import csv 


data_1 = pd.read_csv('weatherAUS.csv')
data_1 = data_1.drop_duplicates()
with open('weatherAUS2.csv', 'w') as csv_file:
    fieldName = ['Rain_Today']
    csvWriter = csv.DictWriter(csv_file, fieldnames=fieldName)
    csvWriter.writeheader()
    for item in data_1.RainToday:
        if item == 'No':
            csvWriter.writerow({'Rain_Today' : 0})
        else:
            csvWriter.writerow({'Rain_Today' : 1})

data_2 = pd.read_csv('weatherAUS2.csv')
data = pd.concat([data_1, data_2], axis = 'columns')
data.to_csv('weather_AUS.csv', index = False)


data['Sunshine'].fillna(0, inplace = True)
data['MaxTemp'].fillna(0, inplace = True)
data['Rainfall'].fillna(0, inplace = True)
data['WindGustSpeed'].fillna(0, inplace = True)
data['WindSpeed9am'].fillna(0, inplace = True)
data['Humidity9am'].fillna(0, inplace = True)
data['Humidity3pm'].fillna(0, inplace = True)
data['Pressure9am'].fillna(0, inplace = True)
data['Cloud9am'].fillna(0, inplace = True)
data['Temp3pm'].fillna(0, inplace = True)
data['RISK_MM'].fillna(0, inplace = True)

rainy = data[data.RainToday == 'Yes']
not_rainy = data[ data.RainToday == 'No']
newData = data.groupby('RainToday').mean()

data = data.drop(['MinTemp', 'Evaporation', 'WindSpeed3pm', 'Pressure3pm', 'Cloud3pm', 'Temp9am', 'Date',
                   'RISK_MM', 'RainTomorrow', 'RainToday'], axis = 'columns')

dummies_Location = pd.get_dummies(data.Location)
dummies_WindGustDir = pd.get_dummies(data.WindGustDir)
dummies_WindDir9am = pd.get_dummies(data.WindDir9am)
dummies_WindDir3pm = pd.get_dummies(data.WindDir3pm)

newData = pd.concat([data, dummies_Location, dummies_WindGustDir, dummies_WindDir9am, dummies_WindDir3pm],
           axis = 'columns')
newData = newData.drop(['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis = 'columns')

x = newData.drop(['Rain_Today'], axis = 'columns')
y = newData.Rain_Today
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)
model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
score = model.score(x_test, y_test)







 

