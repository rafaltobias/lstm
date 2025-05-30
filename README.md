# LSTM Przewidywanie Cen Akcji

Zaawansowany projekt głębokiego uczenia wykorzystujący sieci neuronowe Long Short-Term Memory (LSTM) do przewidywania cen akcji z kompleksową analizą metryk i oceną wydajności.

## 🚀 Funkcje

- **Zaawansowana Architektura LSTM**: Wykorzystuje wielowarstwowe sieci LSTM z dropout do przewidywania cen akcji
- **Dostrajanie Hiperparametrów**: Automatyczna optymalizacja hiperparametrów za pomocą Keras Tuner
- **Walidacja Krzyżowa Szeregów Czasowych**: Solidna walidacja modelu za pomocą TimeSeriesSplit
- **Kompleksowe Metryki**: 
  - Metryki regresji (MAE, MSE, RMSE)
  - Metryki klasyfikacji (Precision, Recall, F1-score dla kierunku zmian cen)
  - Metryki wydajności (czas wykonania, użycie pamięci, rozmiar modelu)
- **Eksport Danych**: Eksport danych treningowych/testowych do CSV do dalszej analizy
- **Wizualizacja**: Krzywe strat i wykresy przewidywań vs rzeczywiste ceny
- **Ochrona przed Rate Limiting**: Solidne pobieranie danych z mechanizmami ponownych prób

## 📋 Wymagania

- Python 3.7+
- TensorFlow 2.8+
- Wymagane biblioteki (zobacz requirements.txt)

## 🛠️ Instalacja

### 1. Klonowanie lub Pobranie Projektu

```bash
# Jeśli używasz git
git clone <url-twojego-repozytorium>
cd lstm

# Lub pobierz pliki bezpośrednio do folderu o nazwie 'lstm'
```

### 2. Utworzenie Środowiska Wirtualnego (Zalecane)

```powershell
# Utwórz środowisko wirtualne
python -m venv lstm_env

# Aktywuj środowisko wirtualne
# Na Windows PowerShell:
.\lstm_env\Scripts\Activate.ps1

# Na Windows Command Prompt:
lstm_env\Scripts\activate.bat

# Na macOS/Linux:
source lstm_env/bin/activate
```

### 3. Instalacja Zależności

```bash
# Zainstaluj wszystkie wymagane pakiety
pip install -r requirements.txt
```

### 4. Alternatywna Instalacja Manualna

Jeśli wolisz zainstalować pakiety ręcznie:

```bash
pip install numpy pandas scikit-learn yfinance tensorflow keras-tuner matplotlib psutil
```

## 🚦 Szybki Start

### Podstawowe Użycie

```python
# Uruchom główny skrypt
python lstm.py
```

Skrypt wykona:
1. Pobierze dane akcji AAPL (lub użyje istniejącego CSV)
2. Przygotuje i wyeksportuje dane treningowe/testowe
3. Wykona dostrajanie hiperparametrów
4. Wytrenuje model LSTM z walidacją krzyżową
5. Wygeneruje przewidywania i kompleksowe metryki
6. Wyeksportuje wyniki do plików CSV

### Dostosowywanie

Edytuj funkcję `main()` w `lstm.py` aby dostosować:

```python
def main():
    ticker = "AAPL"          # Zmień symbol akcji
    start_date = "2015-01-01"  # Dostosuj zakres dat
    end_date = datetime.now().strftime('%Y-%m-%d')
    # ... reszta funkcji
```

## 📊 Pliki Wyjściowe

Skrypt generuje kilka plików wyjściowych:

- `{TICKER}_stock_data.csv` - Surowe dane akcji
- `{TICKER}_X_train.csv` - Cechy treningowe (sekwencje wejściowe LSTM)
- `{TICKER}_y_train.csv` - Cele treningowe
- `{TICKER}_X_test.csv` - Cechy testowe
- `{TICKER}_y_test.csv` - Cele testowe
- `{TICKER}_data_metadata.csv` - Metadane struktury danych
- `{TICKER}_predictions.csv` - Przewidywania modelu vs rzeczywiste ceny
- `{TICKER}_metrics_summary.csv` - Kompleksowe metryki wydajności

## 🏗️ Architektura

### Struktura Modelu
- **Warstwa Wejściowa**: LSTM z konfigurowalnymi jednostkami (64-256)
- **Warstwa Ukryta**: Druga warstwa LSTM z dropout
- **Warstwy Dense**: Pełnie połączone warstwy dla końcowej predykcji
- **Wyjście**: Pojedynczy neuron dla przewidywania ceny

### Kluczowe Komponenty

1. **Przetwarzanie Danych**: Skalowanie MinMax i tworzenie sekwencji
2. **Dostrajanie Hiperparametrów**: Optymalizacja RandomSearch
3. **Walidacja Krzyżowa**: Walidacja uwzględniająca szeregi czasowe
4. **Obliczanie Metryk**: Wielowymiarowa analiza wydajności

## 📈 Wyjaśnienie Metryk

### Metryki Regresji
- **MAE (Mean Absolute Error)**: Średnia bezwzględna różnica między przewidywanymi a rzeczywistymi cenami
- **MSE (Mean Squared Error)**: Średnia kwadratowa różnica (bardziej karze większe błędy)
- **RMSE (Root Mean Squared Error)**: Pierwiastek kwadratowy z MSE, w tych samych jednostkach co cena

### Metryki Klasyfikacji (Kierunek Zmian Cen)
- **Precision**: Dokładność pozytywnych przewidywań (wzrosty cen)
- **Recall**: Zdolność do znalezienia wszystkich pozytywnych przypadków
- **F1-Score**: Średnia harmoniczna precision i recall

### Metryki Wydajności
- **Czas Wykonania**: Czas trenowania i przewidywania
- **Użycie Pamięci**: Zużycie RAM podczas wykonania
- **Rozmiar Modelu**: Ślad pamięciowy sieci neuronowej
- **Przepustowość**: Przewidywania na sekundę

## 🔧 Rozwiązywanie Problemów

### Częste Problemy

1. **Błędy Rate Limiting**
   - Skrypt zawiera mechanizmy ponownych prób dla Yahoo Finance
   - Jeśli problem się utrzymuje, sprawdź połączenie internetowe lub spróbuj później

2. **Problemy z Pamięcią**
   - Zmniejsz `max_trials` w dostrajaniu hiperparametrów
   - Zmniejsz rozmiar okna `look_back`
   - Użyj mniejszych rozmiarów batch

3. **Niewystarczające Dane**
   - Upewnij się, że masz co najmniej 100+ dni danych historycznych
   - Dostosuj `start_date` aby uwzględnić więcej historii

4. **Błędy Importu**
   - Sprawdź czy wszystkie pakiety są zainstalowane: `pip list`
   - Przeinstaluj wymagania: `pip install -r requirements.txt --force-reinstall`

### Optymalizacja Wydajności

- **Przyspieszenie GPU**: Zainstaluj tensorflow-gpu dla szybszego trenowania
- **Zmniejsz Próby**: Obniż `max_trials` dla szybszych wyników
- **Rozmiar Batch**: Eksperymentuj z różnymi rozmiarami batch
- **Early Stopping**: Dostosuj parametr patience

## 🔮 Interpretacja Modelu

### Zrozumienie Przewidywań
- Model przewiduje znormalizowane ceny (zakres 0-1)
- Przewidywania są odwrotnie transformowane do rzeczywistej skali cen
- Dokładność kierunku pokazuje jak dobrze model przewiduje ruchy cen

### Najlepsze Praktyki
- **Jakość Danych**: Zapewnij czyste, spójne dane historyczne
- **Feature Engineering**: Rozważ dodanie wskaźników technicznych
- **Regularne Przekwalifikowanie**: Aktualizuj model o najnowsze dane
- **Zarządzanie Ryzykiem**: Używaj przewidywań jako jeden z czynników w podejmowaniu decyzji

## ⚠️ Zastrzeżenie

Ten projekt jest przeznaczony wyłącznie do celów edukacyjnych i badawczych. Przewidywania giełdowe są z natury niepewne i ten model nie powinien być używany jako jedyna podstawa decyzji finansowych. Zawsze konsultuj się z profesjonalistami finansowymi i przeprowadź własne badania przed podejmowaniem decyzji inwestycyjnych.

## 🙏 Podziękowania

- Zespoły TensorFlow i Keras za framework głębokiego uczenia
- Yahoo Finance za dostarczanie danych giełdowych
- scikit-learn za narzędzia uczenia maszynowego
- Społeczność open-source Python

---