# predicting-sales

This is the code used for my final capstone project as part of my data science program. 

Exploratory data analysis and time series modeling with ARIMA and Facebook Prophet models to predict the sale of perishable grocery store items, with weather data added to improve predictions.

A python jupyter notebook was used for exploratory data analysis and running Facebook Prophet models.

R was used for the ARIMA models to take advantage of the "auto.arima" package that automatically determines the best values for the p (order/number of lags of the autoregressive model), d (degree of differencing), and q (order of the moving-average model) parameters.

Sales data from https://www.kaggle.com/c/favorita-grocery-sales-forecasting

Weather data from https://www.worldweatheronline.com/