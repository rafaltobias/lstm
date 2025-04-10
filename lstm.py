import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras_tuner import RandomSearch

# Funkcja do pobierania danych giełdowych
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Funkcja do zapisu danych do pliku CSV
def save_data_to_csv(data, file_name):
    data.to_csv(file_name, index=False)
    print(f"Dane zostały zapisane do pliku: {file_name}")

# Funkcja do przygotowania danych dla modelu LSTM
def prepare_lstm_data(data, feature_column='Close', look_back=60):
    data = data[[feature_column]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler, scaled_data

# Funkcja budująca model dla hipertuningu
def build_model(hp):
    model = Sequential()
    look_back = 60  # Stały look_back
    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=128, step=32), 
                   return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), return_sequences=False))
    model.add(Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16)))
    model.add(Dense(units=1))
    
    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', 1e-4, 1e-2, sampling='log')),
                  loss='mean_squared_error')
    return model

# Klasa do budowy i trenowania modelu LSTM
class LSTMStockPredictor:
    def __init__(self):
        self.model = None
        self.tuner = None
        self.look_back = 60  # Stały look_back
        self.scaler = None
        self.best_hps = None  # Przechowujemy najlepsze hiperparametry

    def hypertune(self, X_train, y_train, max_trials=25, executions_per_trial=1):
        X_train = np.reshape(X_train, (X_train.shape[0], self.look_back, 1))
        self.tuner = RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='hyperparam_tuning',
            project_name='stock_prediction'
        )
        self.tuner.search(X_train, y_train, epochs=25, validation_split=0.2,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
        
        # Zapisujemy najlepsze hiperparametry zamiast ładować model
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Najlepsze hiperparametry:", self.best_hps.values)

    def train(self, X, y, epochs=25, batch_size=32, n_splits=5, patience=5):
        # Budujemy model na podstawie najlepszych hiperparametrów, jeśli istnieją
        if self.best_hps:
            self.model = Sequential([
                LSTM(units=self.best_hps.get('units_1'), return_sequences=True, input_shape=(self.look_back, 1)),
                Dropout(self.best_hps.get('dropout_1')),
                LSTM(units=self.best_hps.get('units_2'), return_sequences=False),
                Dropout(self.best_hps.get('dropout_2')),
                Dense(units=self.best_hps.get('dense_units')),
                Dense(units=1)
            ])
            self.model.compile(optimizer=Adam(learning_rate=self.best_hps.get('lr')), loss='mean_squared_error')
        else:
            # Domyślny model, jeśli hipertuning nie został wykonany
            self.model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(self.look_back, 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=False),
                Dropout(0.2),
                Dense(units=25),
                Dense(units=1)
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')

        tscv = TimeSeriesSplit(n_splits=n_splits)
        histories = []
        fold_metrics = []

        X = np.reshape(X, (X.shape[0], self.look_back, 1))

        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            print(f"Trenowanie fold {fold + 1}/{n_splits}...")
            X_train_fold, X_val_fold = X[train_index], X[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            history = self.model.fit(
                X_train_fold, y_train_fold,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_fold, y_val_fold),
                callbacks=[early_stopping],
                verbose=1
            )
            histories.append(history)

            val_predictions = self.model.predict(X_val_fold)
            mae = mean_absolute_error(y_val_fold, val_predictions)
            mse = mean_squared_error(y_val_fold, val_predictions)
            rmse = np.sqrt(mse)
            fold_metrics.append((mae, mse, rmse))
            print(f"Fold {fold + 1} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

        avg_mae = np.mean([m[0] for m in fold_metrics])
        avg_mse = np.mean([m[1] for m in fold_metrics])
        avg_rmse = np.mean([m[2] for m in fold_metrics])
        print(f"Średnie wyniki walidacji krzyżowej: MAE: {avg_mae:.4f}, MSE: {avg_mse:.4f}, RMSE: {avg_rmse:.4f}")

        return histories

    def predict(self, X):
        X = np.reshape(X, (X.shape[0], self.look_back, 1))
        return self.model.predict(X)

    def plot_loss(self, histories):
        plt.figure(figsize=(10, 5))
        for i, history in enumerate(histories):
            plt.plot(history.history['loss'], label=f'Fold {i+1} - Loss (Train)')
            plt.plot(history.history['val_loss'], label=f'Fold {i+1} - Loss (Validation)')
        plt.title('Strata treningowa i walidacyjna dla wszystkich foldów')
        plt.xlabel('Epoka')
        plt.ylabel('Strata')
        plt.legend()
        plt.grid()
        plt.show()

# Pozostałe funkcje bez zmian
def plot_predictions(real, predicted, title="Porównanie rzeczywistych cen i przewidywań"):
    plt.figure(figsize=(14, 7))
    plt.plot(real, color='blue', label='Rzeczywiste ceny')
    plt.plot(predicted, color='red', label='Przewidywane ceny')
    plt.title(title)
    plt.xlabel('Czas')
    plt.ylabel('Cena')
    plt.legend()
    plt.grid()
    plt.show()

def calculate_metrics(real, predicted):
    mae = mean_absolute_error(real, predicted)
    mse = mean_squared_error(real, predicted)
    rmse = np.sqrt(mse)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    return mae, mse, rmse

def main():
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    output_file = f"{ticker}_stock_data.csv"

    stock_data = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
    save_data_to_csv(stock_data, output_file)

    print("Przygotowywanie danych dla modelu LSTM...")
    X, y, scaler, scaled_data = prepare_lstm_data(stock_data, feature_column='Close', look_back=60)

    split = int(len(X) * 0.8)
    X_train_full, y_train_full = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    print("Budowanie modelu LSTM z hipertuningiem...")
    lstm_predictor = LSTMStockPredictor()
    lstm_predictor.scaler = scaler
    
    print("Rozpoczynanie hipertuningu...")
    lstm_predictor.hypertune(X_train_full, y_train_full, max_trials=10)

    print("Trenowanie modelu z najlepszymi hiperparametrami i walidacją krzyżową...")
    histories = lstm_predictor.train(X_train_full, y_train_full, epochs=50, batch_size=64, n_splits=5)
    lstm_predictor.plot_loss(histories)

    print("Przewidywanie na danych testowych...")
    predictions = lstm_predictor.predict(X_test)
    predictions = lstm_predictor.scaler.inverse_transform(predictions)

    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    plot_predictions(real_prices, predictions.flatten(), title="Rzeczywiste ceny vs Przewidywania")

    print("Obliczanie skuteczności modelu na zbiorze testowym...")
    mae, mse, rmse = calculate_metrics(real_prices, predictions.flatten())

    print("Przewidywanie ceny na następne 7 dni...")
    last_sequence = scaled_data[-lstm_predictor.look_back:].copy()

    predicted_prices = []
    for _ in range(7):
        current_sequence = np.reshape(last_sequence, (1, lstm_predictor.look_back, 1))
        next_day_prediction = lstm_predictor.predict(current_sequence)
        next_day_prediction = lstm_predictor.scaler.inverse_transform(next_day_prediction)
        predicted_prices.append(next_day_prediction[0][0])
        last_sequence = np.append(last_sequence[1:], lstm_predictor.scaler.transform(next_day_prediction)[0])

    for i, price in enumerate(predicted_prices):
        prediction_date = datetime.now() + timedelta(days=i + 1)
        print(f"Przewidywana cena na {prediction_date.strftime('%Y-%m-%d')}: {price:.2f} USD")

    results = pd.DataFrame({
        "Real": real_prices,
        "Predicted": predictions.flatten()
    })
    results.to_csv(f"{ticker}_predictions.csv", index=False)
    print(f"Wyniki przewidywań zapisano do pliku: {ticker}_predictions.csv")

if __name__ == "__main__":
    main()