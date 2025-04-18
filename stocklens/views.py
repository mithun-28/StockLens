from django.shortcuts import render
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import feedparser
from urllib.parse import quote
from textblob import TextBlob

def home(request):
    context = {
        'symbol': None,
        'stock_data': None,
        'graph': None,
        'prediction': None,
        'news_articles': [],
        'hybrid_table': None,
    }
    return render(request, 'home.html', context)

def safe_info(info, key, default="N/A"):
    return info.get(key, default)

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

def get_stock_data(request):
    symbol = request.GET.get('symbol', '')
    full_symbol = symbol.upper() + '.NS'
    selected_date = request.GET.get('date')

    try:
        stock = yf.Ticker(full_symbol)
        hist = stock.history(period="6mo")

        if hist.empty:
            raise ValueError(f"No historical data found for symbol '{full_symbol}'.")

        info = stock.info
        current_price = info.get('currentPrice', None)

        # Create stock price graph
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Historical Price'))

        price_fig.update_layout(
            title=f"{full_symbol} Stock Price - Last 6 Months",
            xaxis_title='Date',
            yaxis_title='Price (INR)',
            template='plotly_dark'
        )
        graph_html = price_fig.to_html(full_html=False)

        stock_data = {
            'Company Name': safe_info(info, 'shortName'),
            'Sector': safe_info(info, 'sector'),
            'Industry': safe_info(info, 'industry'),
            'Current Price (INR)': current_price,
            'Market Cap': safe_info(info, 'marketCap'),
            'PE Ratio': safe_info(info, 'trailingPE'),
            'Dividend Yield': safe_info(info, 'dividendYield'),
            '52 Week High': safe_info(info, 'fiftyTwoWeekHigh'),
            '52 Week Low': safe_info(info, 'fiftyTwoWeekLow'),
            'Previous Close': safe_info(info, 'previousClose'),
        }

        prediction = None
        news_articles = []

        if selected_date:
            try:
                future_date = datetime.strptime(selected_date, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError("Invalid date format. Please use YYYY-MM-DD.")

            last_date = hist.index[-1].date()
            total_forecast_days = (future_date - last_date).days + 3
            if total_forecast_days <= 0:
                raise ValueError("Selected date must be in the future.")

            weights = np.arange(1, len(hist) + 1)
            wma_price = round(np.dot(hist['Close'], weights) / weights.sum(), 2)

            try:
                arima_model = ARIMA(hist['Close'], order=(5, 1, 0))
                arima_fit = arima_model.fit()
                arima_forecast = arima_fit.forecast(steps=total_forecast_days)
            except Exception as arima_error:
                raise ValueError(f"ARIMA model failed: {arima_error}")

            arima_index = (future_date - last_date).days
            if arima_index >= len(arima_forecast):
                raise ValueError("Forecast range does not cover the selected date.")

            arima_price = round(arima_forecast.iloc[arima_index], 2)
            hybrid_price = round((wma_price + arima_price) / 2, 2)

            wma_advice = "📈 Likely to increase – Consider Buying" if wma_price > current_price else "📉 May drop – Consider Selling"
            arima_advice = "📊 Bullish Trend – Good Outlook" if arima_price > current_price else "⚠️ Bearish Trend – Stay Cautious"
            hybrid_advice = "🧠 Balanced Signal – Monitor Closely" if abs(hybrid_price - current_price) < 10 else (
                "🚀 Strong Uptrend Expected" if hybrid_price > current_price else "📉 Possible Dip Ahead"
            )

            # Multi-day prediction table
            prediction_table = []
            for offset in range(-3, 4):
                day = future_date + timedelta(days=offset)
                index_offset = (day - last_date).days
                if 0 <= index_offset < len(arima_forecast):
                    arima_val = round(arima_forecast.iloc[index_offset], 2)
                    hybrid_val = round((wma_price + arima_val) / 2, 2)
                    prediction_table.append({
                        'date': day.strftime('%Y-%m-%d'),
                        'wma': wma_price,
                        'arima': arima_val,
                        'hybrid': hybrid_val
                    })

            # Prediction graph
            prediction_fig = go.Figure()
            prediction_fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Historical Price'))

            prediction_fig.add_trace(go.Scatter(
                x=[future_date], y=[wma_price],
                mode='markers+text',
                marker=dict(color='orange', size=12),
                name='WMA Prediction',
                text=[f"WMA: ₹{wma_price}"],
                textposition='top center'
            ))

            prediction_fig.add_trace(go.Scatter(
                x=[future_date], y=[arima_price],
                mode='markers+text',
                marker=dict(color='pink', size=12),
                name='ARIMA Prediction',
                text=[f"ARIMA: ₹{arima_price}"],
                textposition='bottom center'
            ))

            prediction_fig.add_trace(go.Scatter(
                x=[future_date], y=[hybrid_price],
                mode='markers+text',
                marker=dict(color='lime', size=14),
                name='Hybrid Prediction',
                text=[f"Hybrid: ₹{hybrid_price}"],
                textposition='middle right'
            ))

            prediction_fig.update_layout(
                title=f"📈 {full_symbol} – Predicted Prices on {future_date}",
                xaxis_title='Date',
                yaxis_title='Price (INR)',
                template='plotly_dark'
            )

            prediction_graph = prediction_fig.to_html(full_html=False)

            prediction = {
                'date': selected_date,
                'wma_price': wma_price,
                'arima_price': arima_price,
                'hybrid_price': hybrid_price,
                'wma_advice': wma_advice,
                'arima_advice': arima_advice,
                'hybrid_advice': hybrid_advice,
                'prediction_graph': prediction_graph,
                'multi_day_predictions': prediction_table
            }

        if info.get('shortName'):
            encoded_company = quote(info.get('shortName'))
            news_url = f"https://news.google.com/rss/search?q={encoded_company}+stock+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(news_url)
            news_articles = [{
                'title': entry.title,
                'link': entry.link,
                'published': entry.published,
                'sentiment': analyze_sentiment(entry.title + " " + entry.get('summary', ''))
            } for entry in feed.entries[:5]]

        context = {
            'symbol': full_symbol,
            'stock_data': stock_data,
            'graph': graph_html,
            'prediction': prediction,
            'news_articles': news_articles,
        }

    except Exception as e:
        context = {'error': f"Could not fetch data for '{symbol}'. Error: {str(e)}"}

    return render(request, 'home.html', context)


def about(request):
    return render(request,'about.html')
