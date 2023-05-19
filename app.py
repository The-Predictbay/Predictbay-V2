from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from pandas import DatetimeIndex
import json
import plotly.graph_objects as go

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
    else:
        ticker = 'AAPL' 

    period = '10y'
    df = yf.download(ticker, period=period)

    closing_prices = df['Close']

    high_value = get_today_high(ticker)
    increase_status_high, percentage_change_high = get_percentage_change_high(ticker)
    close_value = get_today_close(ticker)
    increase_status_Close, percentage_change_Close = get_percentage_change_Close(ticker)
    open_value = get_today_open(ticker)
    increase_status_Open, percentage_change_Open = get_percentage_change_Open(ticker)

    chart_data = [{'x': str(date), 'y': price} for date, price in closing_prices.items()]
    ma100 = closing_prices.rolling(window=100).mean()
    ma100 = [{'x': str(date), 'y': price} for date, price in ma100.items() if not pd.isna(price)]
    ma200 = closing_prices.rolling(window=200).mean()
    ma200 = [{'x': str(date), 'y': price} for date, price in ma200.items() if not pd.isna(price)]

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100: i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Load model
    model = load_model('keras_model.h5')

    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predict = model.predict(x_test)

    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predict = y_predict * scale_factor
    y_test = y_test * scale_factor

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index[int(len(df)*0.70):], y=y_test, name='Original Price'))
    fig2.add_trace(go.Scatter(x=df.index[int(len(df)*0.70):], y=y_predict[:, 0], name='Predict'))
    fig2.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Price',
                    # width=1000,
                    height=500 ,
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
                    )
    graph_html = fig2.to_html(full_html=False)


    last_100_days = data_testing[-100:].values
    scaler = MinMaxScaler()
    last_100_days_scaled = scaler.fit_transform(last_100_days)

    predicted_prices = []

    for i in range(1):
        X_test = np.array([last_100_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_price = model.predict(X_test)
        predicted_prices.append(predicted_price)
        last_100_days_scaled = np.append(last_100_days_scaled, predicted_price)
        last_100_days_scaled = np.delete(last_100_days_scaled, 0)

    predicted_prices = np.array(predicted_prices)
    predicted_prices = predicted_prices.reshape(predicted_prices.shape[0], predicted_prices.shape[2])
    predicted_prices = scaler.inverse_transform(predicted_prices)
    predicted_price = predicted_prices[0][0]

    return render_template('index.html', ticker=ticker, chart_data=chart_data, predicted_price=predicted_price, ma100=ma100,ma200=ma200, graph_html=graph_html,high_value=high_value,close_value=close_value,open_value=open_value,high_status=increase_status_high,high_percent=percentage_change_high,Close_status=increase_status_Close,Close_percent=percentage_change_Close,Open_status=increase_status_Open,Open_percent=percentage_change_Open)

# Function to get today's high value of a stock
def get_today_high(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    if not data.empty:
        return data['High'].iloc[-1]
    return None

# Function to get today's close value of a stock
def get_today_close(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    if not data.empty:
        return data['Close'].iloc[-1]
    return None

# Function to get today's open value of a stock
def get_today_open(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='1d')
    if not data.empty:
        return data['Open'].iloc[-1]
    return None

def get_percentage_change_high(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='2d')
    if len(data) >= 2:
        yesterday_high = data['High'].iloc[-2]
        today_high = data['High'].iloc[-1]
        percentage_change = ((today_high - yesterday_high) / yesterday_high) * 100
        if percentage_change > 0:
            increase_status = 'Increased'
        elif percentage_change < 0:
            increase_status = 'Decreased'
        else:
            increase_status = 'No change'
        return increase_status, percentage_change
    return None, None

def get_percentage_change_Close(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='2d')
    if len(data) >= 2:
        yesterday_high = data['Close'].iloc[-2]
        today_high = data['Close'].iloc[-1]
        percentage_change = ((today_high - yesterday_high) / yesterday_high) * 100
        if percentage_change > 0:
            increase_status = 'Increased'
        elif percentage_change < 0:
            increase_status = 'Decreased'
        else:
            increase_status = 'No change'
        return increase_status, percentage_change
    return None, None

def get_percentage_change_Open(symbol):
    stock = yf.Ticker(symbol)
    data = stock.history(period='2d')
    if len(data) >= 2:
        yesterday_high = data['Open'].iloc[-2]
        today_high = data['Open'].iloc[-1]
        percentage_change = ((today_high - yesterday_high) / yesterday_high) * 100
        if percentage_change > 0:
            increase_status = 'Increased'
        elif percentage_change < 0:
            increase_status = 'Decreased'
        else:
            increase_status = 'No change'
        return increase_status, percentage_change
    return None, None

@app.route('/faq')
def faq():
    return render_template('pages-faq.html')

@app.route('/contact')
def contact():
    return render_template('pages-contact.html')

@app.route('/about')
def about():
    return render_template('pages-about.html')

@app.route('/overview')
def overview():
    return render_template('pages-overview.html')

@app.route('/register')
def register():
    return render_template('pages-register.html')

@app.route('/login')
def login():
    return render_template('pages-login.html')

if __name__ == '__main__':
    app.run(debug=True)
