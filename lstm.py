"""
biblioteki:
pip install numpy pandas yfinance pandas-datareader scikit-learn tensorflow keras-tuner matplotlib psutil

Źródła danych (w kolejności prób):
1. Yahoo Finance (yfinance) - główne źródło
2. IEX (Investors Exchange) - alternatywa dla akcji amerykańskich
3. FRED (Federal Reserve Economic Data) - dla głównych indeksów
4. Stooq - globalne dane giełdowe
5. Yahoo Finance przez pandas_datareader - zapasowa opcja
"""

import numpy as np
import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import time  
import psutil
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras_tuner import RandomSearch

def fetch_stock_data(ticker, start_date, end_date, max_retries=3, delay=5):
    """
    Pobiera dane giełdowe z obsługą błędów rate limiting i alternatywnymi źródłami danych
    """    
    
    for attempt in range(max_retries):
        try:
            print(f"Próba {attempt + 1}/{max_retries} pobrania danych dla {ticker} z Yahoo Finance...")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not data.empty:
                print(f"Pomyślnie pobrano {len(data)} rekordów dla {ticker} z Yahoo Finance")
                return data
            else:
                print(f"UWAGA: Nie pobrano żadnych danych dla {ticker} z Yahoo Finance")
                
        except Exception as e:
            print(f"Błąd podczas pobierania danych z Yahoo Finance (próba {attempt + 1}): {e}")
            
            if "Rate limited" in str(e) or "Too Many Requests" in str(e):
                if attempt < max_retries - 1:
                    wait_time = delay * (2 ** attempt)  
                    print(f"Rate limit detected. Czekanie {wait_time} sekund...")
                    time.sleep(wait_time)
                else:
                    print("Maksymalna liczba prób Yahoo Finance osiągnięta.")
            else:
                print(f"Nieoczekiwany błąd Yahoo Finance: {e}")                
                break
    
    print(f"\nPróbuję alternatywne źródła danych dla {ticker}...")
    
    try:
        print(f"Próba pobrania danych dla {ticker} z IEX...")        
        data = pdr.get_data_iex(ticker, start=start_date, end=end_date)
        if not data.empty:
            if 'close' in data.columns:
                data = data.rename(columns={
                    'open': 'Open', 
                    'high': 'High', 
                    'low': 'Low', 
                    'close': 'Close', 
                    'volume': 'Volume'
                })
            print(f"Pomyślnie pobrano {len(data)} rekordów dla {ticker} z IEX")
            return data
    except Exception as e:        
        print(f"Błąd podczas pobierania danych z IEX: {e}")
    
    try:        
        print(f"Próba pobrania danych dla {ticker} z FRED...")
        fred_symbols = {            
            'SPY': 'SP500',
            'QQQ': 'NASDAQCOM',
            'DIA': 'DJIA',
            'AAPL': None,
            'MSFT': None,
            'GOOGL': None,
            'TSLA': None
        }
        
        fred_symbol = fred_symbols.get(ticker.upper())
        if fred_symbol:            
            data = pdr.get_data_fred(fred_symbol, start=start_date, end=end_date)
            if not data.empty:
                data = pd.DataFrame({
                    'Open': data.iloc[:, 0],                    
                    'High': data.iloc[:, 0], 
                    'Low': data.iloc[:, 0],
                    'Close': data.iloc[:, 0],
                    'Volume': 0
                })
                print(f"Pomyślnie pobrano {len(data)} rekordów dla {ticker} z FRED")
                return data
        else:
            print(f"Symbol {ticker} nie jest dostępny w FRED")
    except Exception as e:        print(f"Błąd podczas pobierania danych z FRED: {e}")
    
    try:
        print(f"Próba pobrania danych dla {ticker} z Stooq...")        
        data = pdr.get_data_stooq(ticker, start=start_date, end=end_date)
        if not data.empty:
            data = data.sort_index()
            print(f"Pomyślnie pobrano {len(data)} rekordów dla {ticker} z Stooq")
            return data
    except Exception as e:        
        print(f"Błąd podczas pobierania danych z Stooq: {e}")
    
    try:
        print(f"Próba pobrania danych dla {ticker} z Yahoo przez pandas_datareader...")
        data = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
        if not data.empty:
            print(f"Pomyślnie pobrano {len(data)} rekordów dla {ticker} z Yahoo przez pandas_datareader")
            return data
    except Exception as e:
            print(f"Błąd podczas pobierania danych z Yahoo przez pandas_datareader: {e}")
    print("BŁĄD: Wszystkie źródła danych zawiodły. Sprawdź połączenie internetowe lub spróbuj z innym tickerem.")
    print("Dostępne alternatywne źródła zostały wyczerpane:")
    print("- Yahoo Finance (yfinance)")
    print("- IEX (Investors Exchange)")
    print("- FRED (Federal Reserve Economic Data)")
    print("- Stooq")
    print("- Yahoo Finance (pandas_datareader)")
    return None

def save_data_to_csv(data, file_name):
    data.to_csv(file_name, index=False)
    print(f"Dane zostały zapisane do pliku: {file_name}")

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

def build_model(hp):
    model = Sequential()
    look_back = 60
    model.add(LSTM(units=hp.Int('units_1', min_value=64, max_value=256, step=64), 
                   return_sequences=True, input_shape=(look_back, 1)))
    model.add(Dropout(hp.Float('dropout_1', 0.2, 0.5, step=0.1)))
    model.add(LSTM(units=hp.Int('units_2', min_value=64, max_value=256, step=64), return_sequences=False))
    model.add(Dropout(hp.Float('dropout_2', 0.2, 0.5, step=0.1)))
    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=128, step=32)))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=hp.Float('lr', 1e-4, 1e-3, sampling='log')),
                  loss='mean_squared_error')
    return model

class LSTMStockPredictor:
    
    def __init__(self):
        self.model = None
        self.tuner = None
        self.look_back = 60 
        self.scaler = None
        self.best_hps = None  

    def hypertune(self, X_train, y_train, max_trials=2, executions_per_trial=1):
        X_train = np.reshape(X_train, (X_train.shape[0], self.look_back, 1))
        self.tuner = RandomSearch(
            build_model,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='hyperparam_tuning',
            project_name='stock_prediction'
        )
        self.tuner.search(X_train, y_train, epochs=50, validation_split=0.2,
                          callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
        
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Najlepsze hiperparametry:", self.best_hps.values)    
    def train(self, X, y, epochs=50, batch_size=64, n_splits=5, patience=10):
        train_start_time = time.time()
        
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
            self.model = Sequential([
                LSTM(units=128, return_sequences=True, input_shape=(self.look_back, 1)),
                Dropout(0.3),
                LSTM(units=128, return_sequences=False),
                Dropout(0.3),
                Dense(units=64),
                Dense(units=1)
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')

        tscv = TimeSeriesSplit(n_splits=n_splits)
        histories = []
        fold_metrics = []

        X = np.reshape(X, (X.shape[0], self.look_back, 1))

        total_start_time = time.time()
        for fold, (train_index, val_index) in enumerate(tscv.split(X)):
            fold_start_time = time.time()
            print(f"\nRozpoczęcie foldu {fold + 1}/{n_splits}...")
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
            
            fold_end_time = time.time()
            fold_duration = fold_end_time - fold_start_time
            print(f"Fold {fold + 1} - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
            print(f"Czas trenowania foldu {fold + 1}: {fold_duration:.2f} sekund")

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        print(f"\nCałkowity czas trenowania: {total_duration:.2f} sekund")
        print(f"Średni czas na fold: {total_duration/n_splits:.2f} sekund")
        
        train_end_time = time.time()
        calculate_efficiency_metrics(train_start_time, train_end_time, get_model_size(self.model))

        return histories    
    def predict(self, X):
        predict_start_time = time.time()
        X = np.reshape(X, (X.shape[0], self.look_back, 1))
        predictions = self.model.predict(X)
        predict_end_time = time.time()
        
        num_predictions = X.shape[0]
        prediction_time = predict_end_time - predict_start_time
        throughput = num_predictions / prediction_time if prediction_time > 0 else 0
        
        print(f"\n=== WYDAJNOŚĆ PREDYKCJI ===")
        print(f"Liczba predykcji: {num_predictions}")
        print(f"Czas predykcji: {prediction_time:.4f} sekund")
        print(f"Throughput: {throughput:.2f} predykcji/sekundę")
        
        return predictions

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

    def predict_next_day(self, data, feature_column='Close'):
        """
        Predicts the next day's stock price using the last look_back days of data
        """
        last_data = data[feature_column].values[-self.look_back:]
        last_data = last_data.reshape(-1, 1)
        
        scaled_data = self.scaler.transform(last_data)
        
        X = np.reshape(scaled_data, (1, self.look_back, 1))
        
        prediction = self.model.predict(X)
        
        next_day_price = self.scaler.inverse_transform(prediction)[0][0]
        
        return next_day_price

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
    print(f"\n=== METRYKI REGRESJI ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    accuracy, threshold_accuracy = calculate_accuracy(real, predicted)
    
    precision, recall, f1, direction_accuracy, cm = calculate_direction_metrics(real, predicted)
    
    return mae, mse, rmse, accuracy, threshold_accuracy, precision, recall, f1, direction_accuracy

def calculate_direction_metrics(real_prices, predicted_prices):
    """
    Oblicza precision, recall, F1 score na podstawie kierunku zmian cen
    """
    if len(real_prices) < 2 or len(predicted_prices) < 2:
        print(f"\n=== METRYKI KIERUNKU ZMIAN ===")
        print("UWAGA: Za mało danych do obliczenia metryk kierunku (minimum 2 próbki)")
        return 0.0, 0.0, 0.0, 0.0, np.array([[0, 0], [0, 0]])
    
    real_directions = np.diff(real_prices) > 0  
    predicted_directions = np.diff(predicted_prices) > 0
    
    unique_real = np.unique(real_directions)
    unique_pred = np.unique(predicted_directions)
    
    print(f"\n=== METRYKI KIERUNKU ZMIAN ===")
    print(f"Liczba próbek kierunków: {len(real_directions)}")
    print(f"Unikalne kierunki rzeczywiste: {unique_real}")
    print(f"Unikalne kierunki przewidywane: {unique_pred}")
    
    if len(unique_real) == 1 or len(unique_pred) == 1:
        print("UWAGA: Wszystkie kierunki są takie same - metryki klasyfikacji mogą być nieprecyzyjne")
        
        direction_accuracy = np.mean(real_directions == predicted_directions) * 100
        print(f"Dokładność kierunku zmian: {direction_accuracy:.2f}%")
        
        if len(unique_real) == 1 and len(unique_pred) == 1 and unique_real[0] == unique_pred[0]:
            return 1.0, 1.0, 1.0, direction_accuracy, np.array([[0, 0], [0, 0]])
        else:
            return 0.0, 0.0, 0.0, direction_accuracy, np.array([[0, 0], [0, 0]])
    
    try:
        precision = precision_score(real_directions, predicted_directions, zero_division=0, average='binary')
        recall = recall_score(real_directions, predicted_directions, zero_division=0, average='binary')
        f1 = f1_score(real_directions, predicted_directions, zero_division=0, average='binary')
        
        cm = confusion_matrix(real_directions, predicted_directions)
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Down    Up")
        print(f"Actual Down   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"Actual Up     {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        direction_accuracy = np.mean(real_directions == predicted_directions) * 100
        print(f"\nDokładność kierunku zmian: {direction_accuracy:.2f}%")
        
        return precision, recall, f1, direction_accuracy, cm
        
    except Exception as e:
        print(f"BŁĄD podczas obliczania metryk kierunku: {e}")
        print("Zwracam wartości domyślne...")
        
        direction_accuracy = np.mean(real_directions == predicted_directions) * 100
        print(f"Dokładność kierunku zmian: {direction_accuracy:.2f}%")
        
        return 0.0, 0.0, 0.0, direction_accuracy, np.array([[0, 0], [0, 0]])

def calculate_efficiency_metrics(start_time, end_time, model_size_mb=None):
    """
    Oblicza metryki wydajności
    """
    execution_time = end_time - start_time
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024 / 1024  
    
    print(f"\n=== METRYKI WYDAJNOŚCI ===")
    print(f"Czas wykonania: {execution_time:.2f} sekund")
    print(f"Wykorzystanie pamięci: {memory_usage_mb:.2f} MB")
    
    if model_size_mb:
        print(f"Rozmiar modelu: {model_size_mb:.2f} MB")
    
    return execution_time, memory_usage_mb

def get_model_size(model):
    """
    Oblicza rozmiar modelu w MB
    """
    total_params = model.count_params()
    size_mb = (total_params * 4) / (1024 * 1024)
    return size_mb

def calculate_accuracy(real, predicted, threshold=0.05):
    """
    Oblicza dokładność procentową przewidywań
    threshold: dopuszczalny margines błędu (5% domyślnie)
    """
    mape = np.mean(np.abs((real - predicted) / real)) * 100
    accuracy = 100 - mape
    
    within_threshold = np.mean(np.abs((real - predicted) / real) <= threshold) * 100
    
    print(f"Średnia dokładność: {accuracy:.2f}%")
    print(f"Przewidywania w marginesie {threshold*100}%: {within_threshold:.2f}%")
    return accuracy, within_threshold

def export_training_data_to_csv(X_train, y_train, X_test, y_test, ticker):
    """
    Eksportuje dane treningowe i testowe do plików CSV
    X_train, X_test: 2D arrays (samples, timesteps) - will be reshaped for LSTM later
    y_train, y_test: 1D arrays
    """    
    print("\n=== EKSPORTOWANIE DANYCH TRENINGOWYCH I TESTOWYCH ===")
    
    y_train_df = pd.DataFrame({
        'y_train': y_train
    })
    y_train_file = f"{ticker}_y_train.csv"
    y_train_df.to_csv(y_train_file, index=False)
    print(f"Dane treningowe Y zapisano do: {y_train_file} ({len(y_train)} próbek)")
    
    y_test_df = pd.DataFrame({
        'y_test': y_test
    })
    y_test_file = f"{ticker}_y_test.csv"
    y_test_df.to_csv(y_test_file, index=False)    
    print(f"Dane testowe Y zapisano do: {y_test_file} ({len(y_test)} próbek)")
    if len(X_train.shape) == 2:
        X_train_2d = X_train
        features_per_timestep = 1
    else:
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        features_per_timestep = X_train.shape[2]
    
    X_train_columns = [f'timestep_{i}' for i in range(X_train_2d.shape[1])]
    X_train_df = pd.DataFrame(X_train_2d, columns=X_train_columns)
    X_train_file = f"{ticker}_X_train.csv"
    X_train_df.to_csv(X_train_file, index=False)    
    print(f"Dane treningowe X zapisano do: {X_train_file} ({X_train.shape[0]} próbek, {X_train.shape[1]} timesteps)")
    
    if len(X_test.shape) == 2:
        X_test_2d = X_test
    else:
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
    
    X_test_columns = [f'timestep_{i}' for i in range(X_test_2d.shape[1])]
    X_test_df = pd.DataFrame(X_test_2d, columns=X_test_columns)
    X_test_file = f"{ticker}_X_test.csv"
    X_test_df.to_csv(X_test_file, index=False)
    print(f"Dane testowe X zapisano do: {X_test_file} ({X_test.shape[0]} próbek, {X_test.shape[1]} timesteps)")
    
    metadata = {
        'original_X_train_shape': str(X_train.shape),
        'original_X_test_shape': str(X_test.shape),
        'y_train_samples': len(y_train),
        'y_test_samples': len(y_test),
        'look_back_window': X_train.shape[1],
        'features_per_timestep': features_per_timestep,
        'export_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_file = f"{ticker}_data_metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)
    print(f"Metadane struktury danych zapisano do: {metadata_file}")
    
    print("Eksport danych treningowych zakończony pomyślnie!")
    return X_train_file, y_train_file, X_test_file, y_test_file, metadata_file

def main():
    ticker = "AAPL"
    start_date = "2015-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    output_file = f"{ticker}_stock_data.csv"

    if os.path.exists(output_file):
        print(f"Znaleziono istniejący plik: {output_file}")
        try:
            stock_data = pd.read_csv(output_file, index_col=0, parse_dates=True)
            print(f"Załadowano {len(stock_data)} rekordów z lokalnego pliku")
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku: {e}")
            print("Próbuję pobrać dane ponownie...")
            stock_data = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
            if stock_data is not None:
                save_data_to_csv(stock_data, output_file)
            else:
                print("BŁĄD: Nie udało się pobrać danych. Sprawdź połączenie lub spróbuj później.")
                return
    else:
        stock_data = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)
        if stock_data is not None:
            save_data_to_csv(stock_data, output_file)
        else:
            print("BŁĄD: Nie udało się pobrać danych. Sprawdź połączenie lub spróbuj później.")
            return

    if len(stock_data) < 100:
        print(f"UWAGA: Za mało danych historycznych ({len(stock_data)} rekordów). Zalecane minimum: 100+")
        return    
    print("Przygotowywanie danych dla modelu LSTM...")
    X, y, scaler, scaled_data = prepare_lstm_data(stock_data, feature_column='Close', look_back=60)

    split = int(len(X) * 0.8)
    X_train_full, y_train_full = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    export_training_data_to_csv(X_train_full, y_train_full, X_test, y_test, ticker)

    print("Budowanie modelu LSTM z hipertuningiem...")
    lstm_predictor = LSTMStockPredictor()
    lstm_predictor.scaler = scaler
    print("Rozpoczynanie hipertuningu...")
    lstm_predictor.hypertune(X_train_full, y_train_full, max_trials=5)  

    print("Trenowanie modelu z najlepszymi hiperparametrami i walidacją krzyżową...")
    histories = lstm_predictor.train(X_train_full, y_train_full, epochs=50, batch_size=64, n_splits=5)
    lstm_predictor.plot_loss(histories)

    print("Przewidywanie na danych testowych...")
    predictions = lstm_predictor.predict(X_test)
    predictions = lstm_predictor.scaler.inverse_transform(predictions)

    real_prices = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    plot_predictions(real_prices, predictions.flatten(), title="Rzeczywiste ceny vs Przewidywania")    
    print("Obliczanie skuteczności modelu na zbiorze testowym...")
    mae, mse, rmse, accuracy, threshold_accuracy, precision, recall, f1, direction_accuracy = calculate_metrics(real_prices, predictions.flatten())
    
    results = pd.DataFrame({
        "Real": real_prices,
        "Predicted": predictions.flatten(),
        "Accuracy": np.abs((real_prices - predictions.flatten()) / real_prices) * 100,
        "Real_Direction": np.diff(np.concatenate([[real_prices[0]], real_prices])) > 0,
        "Pred_Direction": np.diff(np.concatenate([[predictions.flatten()[0]], predictions.flatten()])) > 0
    })
    results.to_csv(f"{ticker}_predictions.csv", index=False)
    print(f"Wyniki przewidywań zapisano do pliku: {ticker}_predictions.csv")
    
    metrics_summary = {
        "MAE": mae,
        "MSE": mse, 
        "RMSE": rmse,
        "Price_Accuracy_%": accuracy,
        "Threshold_Accuracy_%": threshold_accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Direction_Accuracy_%": direction_accuracy,
        "Model_Size_MB": get_model_size(lstm_predictor.model)
    }
    
    metrics_df = pd.DataFrame([metrics_summary])
    metrics_df.to_csv(f"{ticker}_metrics_summary.csv", index=False)
    print(f"Podsumowanie metryk zapisano do pliku: {ticker}_metrics_summary.csv")

    next_day_prediction = lstm_predictor.predict_next_day(stock_data)
    last_price = float(stock_data['Close'].iloc[-1])
    print(f"\nPrzewidywana cena akcji {ticker} na następny dzień: ${next_day_prediction:.2f}")
    print(f"Ostatnia znana cena: ${last_price:.2f}")
    print(f"Różnica: ${(next_day_prediction - last_price):.2f} ({((next_day_prediction/last_price)-1)*100:.2f}%)")

    export_training_data_to_csv(X_train_full, y_train_full, X_test, y_test, ticker)

if __name__ == "__main__":
    main()