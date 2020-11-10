# Introduction - Predicting Sales of Perishable Foods

The goal of this project is to predict the sale of perishable grocery store items using time series analysis, with weather data added to improve predictions.

Python jupyter notebooks were used for data cleaning, exploratory data analysis, and running Facebook Prophet models.

R was used for the ARIMA models to take advantage of the "auto.arima" package that automatically determines the best values for the p (order/number of lags of the autoregressive model), d (degree of differencing), and q (order of the moving-average model) parameters.

# The Data

The source of grocery sales data is from a Kaggle competition (https://www.kaggle.com/c/favorita-grocery-sales-forecasting) where contestants were challenged to accurately predict sales for a large grocery store chain in Ecuador. The dataset originally consisted of 33 different product categories at 54 grocery stores; however, for my project I only focused on the 9 products marked as perishable and sales at 5 stores. To supplement the grocery sales data, I web scraped weather data from World Weather Online (https://www.worldweatheronline.com/) using a previously created API wrapper package (https://github.com/ekapope/WorldWeatherOnline). 

I used 4 separate files from the Kaggle competition to build my final dataset (items.csv, stores.csv, train.csv, and holiday_events.csv). Items contains information on each product available at the grocery stores (4 columns = item number, product category, class, whether perishable; number of rows = 4100). Stores contains information on each of the stores (5 columns = store number, city, state, store type, store cluster; number of rows = 54). Train contains data on daily item sales history (5 columns = date, store number, item number, unit sales, whether on promotion; number of rows = 125,497,040). Holiday events contains information on the days that are holidays in Ecuador (6 columns = date, type, location, location name, description, and transferred; number of rows = 350). I joined train to stores on store number, then items on item number, then holiday events on date. I then joined the weather data (4 columns = date, total precipitation, city, average temperature; number of rows = 8435) to this dataset on date and city. This amounted to 4,179,497 rows of data (daily sales at item level).

My final dataset has 8 columns and 75,011 rows. Each row in the dataset represents the daily sales of one perishable product category at one store. The columns are date (ranging between 2013-01-02 to 2017-08-15), store number (5 unique stores), family (9 unique product categories), unit sales (total daily sales of that product category), on promotion (the number of items in that product category on promotion that day), temperature (average temperature), precipitation (total rainfall), and holiday (whether that day was a holiday or not).

# Preprocessing, Cleaning, and EDA

For preprocessing, I summarized the data set so that daily sales at each store was at the product category level instead of item level, for simplicity. For data cleaning, there was some missing data in the date category and the on promotion column. All stores were missing sales data on Christmas and New Years, likely due to store closures. I therefore deleted these rows. I found that on promotion data was not recorded for any stores at the beginning of the dataset (from January 2013 to April 2014). I consulted the Kaggle website and found a response from the grocery store chain stating that promotions were indeed offered at the time, but data was not recorded. I investigated the on promotion data that existed for each store across years and months for each product category. I found that the average number of items on promotion increased across years and that the average number of items on promotion varied between months. I calculated the daily proportion of items on promotion in each product category at each store. I chose to fill in the missing on promotion data by calculating the proportion of sale items that may have been on promotion based on the fraction of sale items on promotion in the nearest corresponding month at that particular store. 

For EDA, I explored how sales varied by product and found that produce had the highest number of unit sales, followed by dairy, then bread, poultry, meats, deli, eggs, prepared foods, and seafood. I explored how total unit sales varied by store and found that each store had a different number of total unit sales. I explored how the average number of sales of each product category varied between stores and found that the number of sales of each product category was often different between the stores. Because the average sales of each category varied between stores, I determined that it would be best to run separate predictive models for each store. I also found that, at the store level, the average sales of each product category also varied, thus determining that it would be best to also run separate predictive models for each product category. I also found that the average number of sales of each product category was higher on holidays, and varied with average temperature, rainfall, and promotions. This makes it likely that the addition of these external variables (holidays, temperature, rainfall, promotions) will be beneficial to making my time series model more accurate.

# Modeling Completed

For time series predictions, I ran ARIMA models and Facebook Prophet models for each store and product category combination. I evaluated the accuracy of my models using the MAPE on the test set. I found that the average MAPE across all models was 8.7% lower using Facebook Prophet (average MAPE: 16.6%, range: 8.3 - 38.6%) compared to ARIMA (average MAPE: 29.5%, range: 13.3 - 76.5%). Thus, I conclude that Facebook Prophet models fit the data better. 

As each model would likely need individual optimization of hyperparameters, I then went on to put more detailed focus on optimizing/evaluating one Facebook Prophet model for the sale of dairy products at store 37. I found that adjusting some hyperparameters (e.x. reducing the changepoint_prior_scale so that fewer changepoints were detected in the training set) reduced the MAPE from 11.0 % (original model) to 7.0% (uniquely optimized model) (see Graph 1 for a visualization of the model results); however, there was still a prominent section of the test data that the model was not able to predict. I also compared this Prophet model to the simple forecast method of just predicting with the mean value and found my model had a 5.95% lower MAPE (simple mean forecast MAPE = 12.9%). 

As time series predictions can be prone to predicting well in short-term forecasts but not long-term, I also evaluated the performance of my model at predicting forecasts ranging between 37 to 365 days in the future using Facebook Prophet’s cross validation package. I found that the MAPE remained relatively stable whether the model was predicting 37 days in the future or 365 days (average MAPE = 15.9%, range = 8.8 – 23.6%) (see Graph 2). 

Graph 1: The predictions of my optimized Facebook Prophet Model for the test set of sales of dairy at store 37.
 
![Graph1](/images/graph1.png)

Graph 2: The MAPE of my optimized Facebook Prophet Model for sales of dairy at store 37 when forecasting sales 37 to 365 days in the future.

![Graph2](/images/graph2.png)
 
# Findings and Conclusions 

Based on all analysis and modeling of the data, I found that the MAPE for each model varied between stores and products. Of all stores, store 51 had the lowest average MAPE across all product models and store 37 had the highest. Of all products, dairy had the lowest average MAPE across all store models and seafood had the highest. For the model I focused on the most (sales of dairy at store 37), I met my goal of creating a time series model that had a lower MAPE than the simple forecast method of just predicting the mean. More work is needed to uniquely optimize/evaluate the other store and product combination models.

# Final Summary

Overall, I have created the framework for further development of this model for predicting sales of perishable foods at grocery stores. These grocery stores could use my model to forecast demand for the perishable food categories and place orders/stock shelves accordingly. For next steps, I would like to try to strengthen the approach by using the stronger model of recurrent neural networks (RNN). I was able to briefly run a RNN model for dairy sales at store 37 and found that the RNN model had a 1.9% lower MAPE than the Facebook Prophet model.