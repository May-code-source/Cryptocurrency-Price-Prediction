# Cryptocurrency-Price-Prediction
## Table of content

- [Project Overview](#project-overview)
- [Highlight](#highlight)
- [Data Source](#data-source)
- [Data Preprocessing Tools and Techniques](#data-preprocessing-tools-and-techniques)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Evaluation](#model-evaluation)
- [Insights](#insights)
- [Considerations ](#considerations)


## Project Overview

Investors seek accurate cryptocurrency price predictions to maximize profits in this volatile market. This project involved predicting cryptocurrency prices using machine learning models. The goal was to forecast prices across different time horizons to aid investment decisions.

## Highlight
- Data collection using Binance API
- Exploratory data analysis
- Feature engineering
- Modeling with LSTM, XGBoost, Random Forest, and evaluation
- GUI development with Streamlit

## Data Source

Historical daily price data from Binance API for the top 20 cryptocurrencies from 2017-2023. The dataset included various market data variables such as open, high, low, close prices, volume, and more.

## Data Preprocessing Tools and Techniques

- Python
- Pandas, Matplotlib, Seaborn for EDA
- Sklearn for preprocessing
- Statistical Analysis
- Time series analysis
- Correlation Analysis 
- Feature Engineering
- Normalization  
- Decision Tree, XGBoost, Random Forest for predictive modeling 
- Streamlit for Graphical User Interface

### Exploratory Data Analysis:
Data was examined for patterns, trends, and outliers using statistical analysis and visualizations

### Model Evaluation:
Performance metrics, including Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Root Mean Squared Error (RMSE), and R-squared (R2), were calculated for model evaluation.

![Picture1](https://github.com/May-code-source/Cryptocurrency-Price-Prediction/assets/115402970/bf47e91f-fba3-433c-b8d8-2311785011da)


## Insights

- Most coins exhibited positive skewness in their price distribution. Positive skewness indicates that the majority of cryptocurrencies tend to have prices skewed towards higher values, this suggests potential for sharp upward movements in prices. This indicates investors may benefit from long positions.
- High volatility observed in prices for all cryptocurrencies  indicates frequent and significant price fluctuations in the cryptocurrency market. This can present opportunities for traders but also increased risks. The need for risk management strategies is required to minimize the downside.
- There is a significant positive correlation found between Bitcoin and other major coins like Ethereum. Investors in altcoins should be aware that the performance of these coins can be influenced by Bitcoin's price. Diversifying across different coins may not always provide complete independence from Bitcoin's price trends
- Feature importance analysis revealed that open, high and low prices were top predictors
- Short-term predictions were most accurate, with decreasing accuracy for longer horizons
- Random Forest outperformed Decision Tree and XGBoost, achieving the best R2 score. The model provided reliable short and medium-term predictions.

## Considerations  
- Model selection and hyperparameter tuning required time and experimentation.
- Feature engineering involved trial and error.
- Balancing model complexity and generalization to avoid overfitting or underfitting required careful consideration..
