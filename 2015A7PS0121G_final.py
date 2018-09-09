import numpy as np
import pandas as pd
import sklearn
import sys

arguments = sys.argv[1:]

data=pd.read_csv(arguments[0])


test=pd.read_csv(arguments[1])

cols_to_drop = [
    'Id','Category']

df = data.drop(cols_to_drop, axis=1)

y=data['Category']


df = pd.concat([df, pd.get_dummies(df['PdDistrict']).rename(columns=lambda x: str(x))], axis=1)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['X', 'Y']] = scaler.fit_transform(df[['X', 'Y']])



test[['X', 'Y']] = scaler.fit_transform(test[['X', 'Y']])



test = pd.concat([test, pd.get_dummies(test['PdDistrict']).rename(columns=lambda x: str(x))], axis=1)



cols_to_drop = [
    'PdDistrict','Address','Descript']

df = df.drop(cols_to_drop, axis=1)


cols_to_drop = [
    'PdDistrict','Address']
test = test.drop(cols_to_drop, axis=1)


df = pd.concat([df, pd.get_dummies(df['DayOfWeek']).rename(columns=lambda x: str(x))], axis=1)
test = pd.concat([test, pd.get_dummies(test['DayOfWeek']).rename(columns=lambda x: str(x))], axis=1)


df = df.drop('DayOfWeek', axis=1)
test = test.drop('DayOfWeek', axis=1)



df['Dates']=pd.to_datetime(df['Dates'])


from datetime import datetime
import time
df['Dates'] = df['Dates'].apply(lambda x: time.mktime(x.timetuple()))



test['Dates']=pd.to_datetime(test['Dates'])

test['Dates'] = test['Dates'].apply(lambda x: time.mktime(x.timetuple()))


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing


le = preprocessing.LabelEncoder()
le.fit(df['Resolution'])



df['Resolution'] = le.fit_transform(df['Resolution']) 


test['Resolution'] = le.fit_transform(test['Resolution']) 

test[['Dates']] = scaler.fit_transform(test[['Dates']])
df[['Dates']] = scaler.fit_transform(df[['Dates']])



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
crime =le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(df, crime, test_size=0.20,
                                                    random_state=42)


from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier



from sklearn.tree import DecisionTreeClassifier
clf4 = DecisionTreeClassifier(max_depth = 6,criterion="entropy")
clf4.fit(X_train, y_train)
pred = clf4.predict_proba(X_test)
l=log_loss(y_test, pred, labels=[0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38])
#print(l)

prediction = clf4.predict_proba(test.drop('Id',axis=1))



result=pd.concat([test['Id'],pd.DataFrame(prediction)],axis=1)
result.columns= ['Id','ARSON','ASSAULT','BAD CHECKS','BRIBERY','BURGLARY','DISORDERLY CONDUCT','DRIVING UNDER THE INFLUENCE','DRUG/NARCOTIC','DRUNKENNESS','EMBEZZLEMENT','EXTORTION','FAMILY OFFENSES','FORGERY/COUNTERFEITING','FRAUD','GAMBLING','KIDNAPPING',
'LARCENY/THEFT','LIQUOR LAWS','LOITERING','MISSING PERSON','NON-CRIMINAL','OTHER OFFENSES','PORNOGRAPHY/OBSCENE MAT','PROSTITUTION','RECOVERED VEHICLE','ROBBERY','RUNAWAY','SECONDARY CODES','SEX OFFENSES FORCIBLE','SEX OFFENSES NON FORCIBLE','STOLEN PROPERTY','SUICIDE','SUSPICIOUS OCC','TREA','TRESPASS','VANDALISM','VEHICLE THEFT','WARRANTS','WEAPON LAWS']


result.to_csv("2015A7PS0121G.csv",index = False)

