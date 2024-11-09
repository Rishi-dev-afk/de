import streamlit as st
import torch
import yfinance as yf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def load_and_clean_data(stock_symbols, start_date, end_date):
    data_list = []
    for symbol in stock_symbols:
        data = yf.download(symbol, start=start_date, end=end_date)
        data = data[['Close']].rename(columns={'Close': symbol})
        data_list.append(data)
    merged_data = pd.concat(data_list, axis=1)
    merged_data.dropna(inplace=True)
    return merged_data

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def create_sequences(data, seq_length=60):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(torch.tensor(data[i:i+seq_length], dtype=torch.float32))
        y.append(torch.tensor(data[i+seq_length], dtype=torch.float32))
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y

class StockLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=4, output_size=3, dropout=0.3):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.batch_norm(out[:, -1, :])
        out = self.fc(self.dropout(out))
        return out

def predict_next_day(model, input_sequence, scaler,device):
    model.eval()
    with torch.no_grad():
        input_sequence = input_sequence.unsqueeze(0).to(device) 
        prediction = model(input_sequence)
        prediction = scaler.inverse_transform(prediction.cpu().numpy())  
        return prediction[0]


def app():
    st.title("Stock Price Prediction using LSTM")

    
    stock_symbols = st.text_input("Enter stock symbols (comma separated)", "AAPL, MSFT, GOOGL")
    stock_symbols = stock_symbols.split(",") 
    start_date = st.date_input("Start Date", value=pd.to_datetime("2008-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    seq_length = st.slider("Sequence Length", min_value=30, max_value=120, value=60, step=10)
    data = load_and_clean_data(stock_symbols, str(start_date), str(end_date))
    st.write("Data loaded successfully!")
    normalized_data, scaler = normalize_data(data)
    x, y = create_sequences(normalized_data, seq_length=seq_length)
    train_size = int(len(x) * 0.8)
    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(stock_symbols)  
    output_size = 3  
    model = StockLSTM(input_size=input_size, hidden_size=128, num_layers=4, output_size=output_size).to(device)

    model.load_state_dict(torch.load("trained_stock_lstm_model.pth", map_location=device))
    st.write("Pretrained model loaded successfully!")

    last_sequence = test_x[-1]
    predicted_prices = predict_next_day(model, last_sequence, scaler, device)
    for i, symbol in enumerate(stock_symbols):
        st.write(f"Predicted next price for {symbol}: ${predicted_prices[i]:.2f}")  
        
    predictions, actuals = get_predictions(model, test_loader, scaler, stock_symbols)
    plot_predictions(predictions, actuals, stock_symbols)

def get_predictions(model, test_loader, scaler, stock_symbols):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(actuals)
    return predictions, actuals

def plot_predictions(predictions, actuals, stock_symbols):
    if not 'stock_symbols' in locals():
        st.error("Stock symbols not found for plotting.")
        return
    plt.figure(figsize=(14, 7))
    for i, symbol in enumerate(stock_symbols):
        plt.plot(actuals[:, i], label=f"Real Price ({symbol})", linestyle='--')
        plt.plot(predictions[:, i], label=f"Predicted Price ({symbol})")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.title("Predicted vs Real Stock Prices")
    st.pyplot(plt)

if __name__ == "__main__":
    app()
