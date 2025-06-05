import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use("macosx")
from keras.models import Sequential
from keras.layers import LSTM, Dense

# 1. Wczytanie i przygotowanie danych
# Uwaga: jeśli używasz pliku 'alta_to_2025-05-11_akcje.xls', zmień nazwę poniżej
df = pd.read_excel('alta.xls', engine='xlrd')
df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)

# 1.a) Rozbicie kolumny 'Data' na Rok, Miesiąc, Dzień
df['Rok'] = df['Data'].dt.year
df['Miesiąc'] = df['Data'].dt.month
df['Dzień'] = df['Data'].dt.day

# 1.b) Najpierw próbujemy znaleźć grupę rok–miesiąc z 100–1000 notowań
counts_month = (
    df
    .groupby(['Rok', 'Miesiąc'])
    .size()
    .reset_index(name='IleNotowan')
)

dobry_okres = counts_month[
    (counts_month['IleNotowan'] >= 100) &
    (counts_month['IleNotowan'] <= 1000)
].copy()

if not dobry_okres.empty:
    # Jeśli istnieje co najmniej jedna para (rok, miesiąc) z 100–1000 notowań:
    rok_wybr  = int(dobry_okres.iloc[0]['Rok'])
    mies_wybr = int(dobry_okres.iloc[0]['Miesiąc'])
    df_seg = df[(df['Rok'] == rok_wybr) & (df['Miesiąc'] == mies_wybr)].copy()
    print(f"Wybrano grupę rok–miesiąc: {rok_wybr}-{mies_wybr} (liczba notowań = {len(df_seg)})")
else:
    # Jeśli nie ma żadnej grupy miesiąc z 100–1000 notowań, grupujemy tylko po roku
    counts_year = (
        df
        .groupby(['Rok'])
        .size()
        .reset_index(name='IleNotowan')
    )
    dobry_roki = counts_year[
        (counts_year['IleNotowan'] >= 100) &
        (counts_year['IleNotowan'] <= 1000)
    ].copy()
    if not dobry_roki.empty:
        rok_wybr = int(dobry_roki.iloc[0]['Rok'])
        df_seg = df[df['Rok'] == rok_wybr].copy()
        print(f"Brak miesiąca 100–1000, wybrano rok: {rok_wybr} (liczba notowań = {len(df_seg)})")
    else:
        # Jeśli nawet rok nie ma 100–1000 notowań, bierzemy ostatnie 500 wierszy (przykładowy fallback)
        df_seg = df.tail(500).copy()
        print(f"Brak roku ani miesiąca w przedziale 100–1000, wybrano ostatnie 500 wierszy (liczba notowań = {len(df_seg)})")

# 1.c) Wybór kolumny 'Kurs zamknięcia'
data_seg = df_seg['Kurs zamknięcia'].values.reshape(-1, 1)

# Skalowanie (MinMax)
scaler = MinMaxScaler()
data_seg_scaled = scaler.fit_transform(data_seg).flatten()

# ------------------------------------------------------------------------------
# 2. MODEL AR
# ------------------------------------------------------------------------------

def create_lagged_data(series: np.ndarray, p: int):
    """
    Buduje X_ar i y_ar:
    - X_ar: macierz, gdzie każdy wiersz to p poprzednich wartości
    - y_ar: odpowiadająca "następna" wartość po tych p
    Argumenty:
      series: 1D array (zeskalowany)
      p: liczba opóźnień
    """
    X_ar, y_ar = [], []
    for i in range(p, len(series)):
        X_ar.append(series[i-p:i])
        y_ar.append(series[i])
    return np.array(X_ar), np.array(y_ar)

# Testujemy p = 5, 10, 15, …, 50
lags = list(range(5, 51, 5))
ar_errors = []

for p in lags:
    X_ar, y_ar = create_lagged_data(data_seg_scaled, p)
    split_ar = int(0.8 * len(X_ar))
    X_train, X_test, y_train, y_test = train_test_split(X_ar, y_ar, test_size=0.2)
    # Uzupełniamy o kolumnę jedynek (intercept)
    X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    beta = np.linalg.pinv(X_train_aug) @ y_train
    X_test_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    y_pred_ar = X_test_aug @ beta

    mse_ar = mean_squared_error(y_test, y_pred_ar)
    ar_errors.append(mse_ar)

# Wybór najlepszego p dla AR
best_ar_idx = np.argmin(ar_errors)
best_p_ar = lags[best_ar_idx]
print(f'Najlepszy model AR: p = {best_p_ar}, MSE (skala [0,1]) = {ar_errors[best_ar_idx]:.6f}')

# Wykres błędów AR vs p
plt.figure(figsize=(10, 5))
plt.plot(lags, ar_errors, marker='o', linestyle='--', color='b', label='AR MSE (skala [0,1])')
plt.xlabel('Liczba opóźnień (p)')
plt.ylabel('MSE (skalowane)')
plt.title('Model AR – Błąd MSE dla różnych opóźnień')
plt.legend()
plt.grid()
plt.show()

# ------------------------------------------------------------------------------
# 3. MODEL LSTM
# ------------------------------------------------------------------------------

def create_sequences(data: np.ndarray, n_steps: int):
    """
    Buduje X_seq i y_seq:
    - X_seq: tablica o kształcie [liczba_próbek, n_steps, 1]
    - y_seq: odpowiadająca nast. wartość (jednowymiarowa)
    Argumenty:
      data: 1D array (zeskalowany)
      n_steps: długość sekwencji (ilość opóźnień)
    """
    X_seq, y_seq = [], []
    for i in range(len(data) - n_steps):
        X_seq.append(data[i:i + n_steps])
        y_seq.append(data[i + n_steps])
    return np.array(X_seq), np.array(y_seq)

lstm_errors = []

for p in lags:
    # 3.a) Tworzymy sekwencje
    X_seq, y_seq = create_sequences(data_seg_scaled.reshape(-1, 1), p)
    split_lstm = int(0.8 * len(X_seq))

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2)
    model = Sequential()
    model.add(LSTM(100, input_shape=(p, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 3.c) Trenowanie
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    model.save("LSTM_model.keras")
    # 3.d) Predykcja i obliczenie MSE w skali oryginalnej
    y_pred_seq = model.predict(X_test)
    # Odwracamy MinMaxScaler do skali faktycznej
    y_pred_resc = scaler.inverse_transform(y_pred_seq)
    y_test_resc = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse_lstm = mean_squared_error(y_test_resc, y_pred_resc)
    lstm_errors.append(mse_lstm)

# Wybór najlepszego p dla LSTM
best_lstm_idx = np.argmin(lstm_errors)
best_p_lstm = lags[best_lstm_idx]
print(f'Najlepszy model LSTM: p = {best_p_lstm}, MSE (oryginalna skala) = {lstm_errors[best_lstm_idx]:.6f}')

# Wykres błędów LSTM vs p (oryginalna skala)
plt.figure(figsize=(10, 5))
plt.plot(lags, lstm_errors, marker='o', linestyle='--', color='magenta', label='LSTM MSE (oryg. skala)')
plt.xlabel('Liczba opóźnień (p)')
plt.ylabel('MSE (skala kursu)')
plt.title('Model LSTM – Błąd MSE dla różnych opóźnień')
plt.legend()
plt.grid()
plt.show()
