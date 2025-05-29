import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_excel('alta.xls')
df['Data'] = pd.to_datetime(df['Data'])

data = df['Zamkniecie'].values.reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Columns
df['rok'] = df['Data'].dt.year
df['miesiac'] = df['Data'].dt.month
df['dzien'] = df['Data'].dt.day
X_train, X_test, y_train, y_test = train_test_split(df, test_size = 0.2, random_state = 42)

#c = np.hstack([x, np.ones(y.shape)])
c = X_train.columns
v = np.linalg.pinv(c) @ y_train

print(v)
def generate_model():
    model = Sequential()
    model.add(LSTM(100, input_shape=(3, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
    return model

y1 = generate_model().predict(X_test)

plt.plot(y_test, 'r-')
plt.plot(y1, 'k-')
plt.plot(v[0]*X_test[:,0] + v[1]*X_test[:,1] + v[2]*X_test[:,2], 'g-')
plt.plot(X_test[:,0], 'b-')
plt.show()
