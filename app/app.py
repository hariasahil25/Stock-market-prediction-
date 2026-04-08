from flask import Flask, render_template, request, jsonify
import numpy as np
import yfinance as yf
import os

app = Flask(__name__)

# Dummy model fallback (for deployment safety)
model = None

def get_model():
    global model
    if model is None:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense

        model = Sequential()
        model.add(LSTM(50, input_shape=(60,1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
    return model


@app.route("/")
def home():
    return render_template("index.html")


# 📈 Real-time stock data API
@app.route("/stock-data")
def stock_data():
    ticker = request.args.get("ticker", "AAPL")

    df = yf.download(ticker, period="1mo", interval="1d")

    return jsonify({
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "prices": df['Close'].fillna(0).tolist()
    })


# 🤖 Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    data = np.array(values).reshape(1, len(values), 1)

    model = get_model()
    prediction = model.predict(data)

    return render_template("index.html", prediction=prediction[0][0])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
