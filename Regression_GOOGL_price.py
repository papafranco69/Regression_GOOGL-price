
import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn import model_selection
from sklearn.linear_model import LinearRegression


quandl.ApiConfig.api_key = "KxMbLvft3VdLU7yKgej1"
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#Find relations between High/ Low, Open/ Close prices
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Open'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100.0

#new features
df = df[['Adj. Close','HL_PCT', 'PCT_change', 'Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

#predict 1% of the data frame
forecast_out = math.ceil(0.01*len(df))

#shifted values of Adjusted Close UP(-) 34 ROWS. 
#In this working example each ROW is equivalent to ONE DAY

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
x = preprocessing.scale(x)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size =0.2)

clf = LinearRegression()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print(accuracy)