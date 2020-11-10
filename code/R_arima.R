# Capstone ARIMA models
install.packages("rmarkdown")
install.packages('forecast')
install.packages('Metrics')
install.packages('tseries')
install.packages('lubridate')
library(lubridate)
library(forecast)
library(Metrics)
library(tseries)
library(rmarkdown)

# reading data
final_new = read.csv('/Users/paigemcdonald/Desktop/final_new.csv')

#check data types
sapply(final_new, class)

#change date to datetime to separate into train and test set based on year
final_new$date <- ymd(as.character(final_new$date))

# split into training set (years 2013-2016) and test set (year 2017)
trainall <- subset(final_new, date < '2017-01-01')
testall <- subset(final_new, date >= '2017-01-01')

# create list of stores and family to loop over
stores <- unique(c(final_new$store_nbr))
family <- unique(c(as.character(final_new$family)))

# for loop over stores and families to run ARIMA models
mapes <- c(as.numeric()) # store results here to calculate average MAPE
for (store in stores) {
    for (product in family) {
      cat(store, product)
      train = subset(trainall, ((store_nbr == store) & (family == product)))
      test = subset(testall, ((store_nbr == store) & (family == product)))
      # Separate out y 
      y_train = train[,c("unit_sales")]
      y_test = test[,c("unit_sales")]
      # Separate out exogenous regressors
      X_reg_train = train[,c("holiday","avgtemp","precipMM","onpromotion")]
      X_reg_test = test[,c("holiday","avgtemp","precipMM","onpromotion")]
      #convert to matrices
      X_reg_train_matrix = data.matrix(X_reg_train)
      X_reg_test_matrix = data.matrix(X_reg_test)
      # training model
      model = auto.arima(y_train, xreg = X_reg_train_matrix)
      # model summary
      summary(model)
      # forecasting
      forecast = predict(model,length(y_test),newxreg = X_reg_test_matrix)
      # evaluation of MAE
      mape_val = mape(y_test,forecast$pred)
      cat('MAPE:', mape(y_test,forecast$pred))
      summary(forecast)
      mapes = append(mapes, mape_val)
    }
}

# calculate average MAPE to compare to Facebook Prophet
summary(mapes)

# Result = average MAPE of auto ARIMA models is 29.5%