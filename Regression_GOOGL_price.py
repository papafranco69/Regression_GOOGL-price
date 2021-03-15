import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


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

x = np.array(df.drop(['label'], 1))
x = preprocessing.scale(x)

x_lately = x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size =0.2)

clf = LinearRegression(n_jobs = 1)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

#an array of forecasts
forecast_set = clf.predict(x_lately)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
# one_day is 86,400 seconds
one_day = 86400
next_unix = last_unix + one_day

# iterating through the forecast set, taking each forecast and day, 
#and setting those as values in the data frame,

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day

#next day is the index of the data frame
df.loc[next_date] = [np.nan for j in range(len(df.columns)-1)] + [i]
print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc =4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()