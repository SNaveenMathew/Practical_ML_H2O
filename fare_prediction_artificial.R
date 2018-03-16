library(h2o)

h2o.init()
set.seed(1) # For reproducibility

# Step 1: Creating a data set of 500 examples
N <- 500

d <- data.frame(id = 1:N)
d$age <- runif(N, min = 4, max = 75)

v <- round(rnorm(N, mean = 1, sd = 0.5))
v <- v + ifelse(d$age > 65, 4, ifelse(d$age > 35, 3, ifelse(d$age > 10, 2, 1)))
v <- pmax(v, 3)
v <- pmin(v, 5)
d$Pclass <- as.factor(v - 2)

v <- 15 + (d$age^2)*((d$Pclass=="1")*5 + (d$Pclass=="2")*10 +
                       (d$Pclass=="3")*15)/1000
v <- v + (d$age) * 1.5
v <- v + runif(N, 0, 25)
d$Fare <- round(v, -1) # Rounding to nearest $10

as.h2o(d, destination_frame = "fare_pred")

# Step 2: Importing data
d <- h2o.getFrame("fare_pred")

# Step 3: Performing train, validation, test split
parts <- h2o.splitFrame(
  d, c(0.7, 0.15), destination_frames = c("fare_train", "fare_valid", "fare_test"))
train <- parts[[1]]
valid <- parts[[2]]
test <- parts[[3]]

# Step 4: Building GBM model on train data and evaluating performance
gbm_model <- h2o.gbm(x = 2:3, y = 4, training_frame = train,
                     validation_frame = valid)
test_performance <- h2o.performance(model = gbm_model, newdata = test)
print(test_performance)
"MSE:  72.33594 | RMSE:  8.505054 | MAE:  6.829138 | RMSLE:  0.1172645 | Mean Residual Deviance :  72.33594"

valid_performance <- h2o.performance(model = gbm_model, newdata = valid)
print(valid_performance)
"MSE:  80.63908 | RMSE:  8.979927 | MAE:  7.740212 | RMSLE:  0.1194677 | Mean Residual Deviance :  80.63908"

# Step 5: Building another GBM model to show performance difference
gbm_model2 <- h2o.gbm(x = 2:3, y = 4, training_frame = train,
                      validation_frame = valid, ntrees = 500)
test_performance2 <- h2o.performance(model = gbm_model2, newdata = test)
print(test_performance2)
"MSE:  76.83609 | RMSE:  8.76562 | MAE:  7.013866 | RMSLE:  0.1197506 | Mean Residual Deviance :  76.83609"

valid_performance2 <- h2o.performance(model = gbm_model2, newdata = valid)
print(valid_performance2)
"MSE:  86.16417 | RMSE:  9.282466 | MAE:  7.77701 | RMSLE:  0.1207156 | Mean Residual Deviance :  86.16417"
# Reproducibility checked by rerunning script after shutting down h2o

h2o.shutdown()
