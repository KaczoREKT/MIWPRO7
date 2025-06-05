from tensorflow.keras.models import load_model

model = load_model("LSTM_model.h5")
mse = model.evaluate()