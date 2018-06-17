# Loading package, importing file and understanding the data set
library(h2o)
h2o.init(max_mem_size = "6g")
in_data <- h2o.importFile("http://coursera.h2o.ai/house_data.3487.csv")

# Converting few columns to factor. Converting bedroom and floors to factor
# because they are not effective as continuous variables
in_data[, "zipcode"] <- h2o.asfactor(in_data[, "zipcode"])
in_data[, "yr_built"] <- h2o.asfactor(in_data[, "yr_built"])
in_data[, "yr_renovated"] <- h2o.asfactor(in_data[, "yr_renovated"])
in_data[, "waterfront"] <- h2o.asfactor(in_data[, "waterfront"])
in_data[, "view"] <- h2o.asfactor(in_data[, "view"])
in_data[, "grade"] <- h2o.asfactor(in_data[, "grade"])
in_data[, "condition"] <- h2o.asfactor(in_data[, "condition"])
in_data[, "floors"] <- h2o.asfactor(in_data[, "floors"])
in_data[, "bedrooms"] <- h2o.asfactor(in_data[, "bedrooms"])

# Doing some feature engineering: year, month, year-month, date as continuous,
# quantiles of latitude and longitude
in_data <- h2o.cbind(
  in_data, as.h2o(data.frame(assessment_year = substr(as.vector(in_data[, "date"]),
                                                      start = 1, stop = 4))))
in_data <- h2o.cbind(
  in_data, as.h2o(data.frame(assessment_month = substr(as.vector(in_data[, "date"]),
                                                       start = 5, stop = 6))))
in_data <- h2o.cbind(
  in_data, as.h2o(data.frame(assessment_year_month = paste0(
    substr(as.vector(in_data[, "date"]), start = 1, stop = 4),
    substr(as.vector(in_data[, "date"]), start = 5, stop = 6)))))
in_data <- h2o.cbind(in_data, as.h2o(
  data.frame(julian_date = as.integer(as.Date(paste0(substr(as.vector(
    in_data[, "date"]), start = 1, stop = 4), "/", substr(as.vector(
      in_data[, "date"]), start = 5, stop = 6), "/", substr(as.vector(
        in_data[, "date"]), start = 7, stop = 8)))))))
lat_qtls <- quantile(as.vector(in_data[, "lat"]), seq(0,1,by=1/250))
lat_cuts <- cut(as.vector(in_data[, "lat"]), breaks = lat_qtls)
lat_cuts[is.na(lat_cuts)] <- "(47.643,47.6453]"
in_data <- h2o.cbind(in_data, as.h2o(data.frame(lat_range = lat_cuts)))
long_qtls <- quantile(as.vector(in_data[, "long"]), seq(0,1,by=1/200))
long_cuts <- cut(as.vector(in_data[, "long"]), breaks = long_qtls)
long_cuts[is.na(long_cuts)] <- "(-122.519,-122.43]"
in_data <- h2o.cbind(in_data, as.h2o(data.frame(long_range = long_cuts)))
summary(in_data)

# Splitting data for machine learning and validation
parts <- h2o.splitFrame(in_data, 0.9, seed = 123)
train <- parts[[1]]
test <- parts[[2]]
nrow(train)
#[1] 19462 -> Verified
parts <- h2o.splitFrame(train, 0.7, seed = 123)
train <- parts[[1]]
valid <- parts[[2]]

# I used h2o.automl to identify top 4 performing models.
# I used the top 4 models with tuned parameters here.
model1 <- h2o.gbm(x = c(4:17, 20:27), y = 3, training_frame = train, nfolds = 5,
                  validation_frame = valid, seed = 1,
                  keep_cross_validation_predictions = T,
                  score_tree_interval = 5, fold_assignment = "Modulo",
                  ntrees = 149, max_depth = 3, stopping_tolerance = 0.00855107,
                  learn_rate = 0.08, distribution = "gaussian",
                  col_sample_rate_per_tree = 0.4, min_split_improvement = 1e-04)
h2o.saveModel(model1, "model1_gbm1.h2omodel")
model2 <- h2o.deeplearning(x = c(4:17, 20:27), y = 3, nfolds = 5,
                           training_frame = train,
                           keep_cross_validation_predictions = T,
                           fold_assignment = "Modulo",
                           overwrite_with_best_model = F, hidden = c(10, 10, 10),
                           epochs = 10.39328, seed = 1, stopping_rounds = 0,
                           stopping_tolerance = 0.00855107,
                           validation_frame = valid)
h2o.saveModel(model2, "model2_dl1.h2omodel")
model3 <- h2o.gbm(x = c(4:17, 20:27), y = 3, training_frame = train, nfolds = 5,
                  validation_frame = valid, seed = 1,
                  keep_cross_validation_predictions = T,
                  score_tree_interval = 5, fold_assignment = "Modulo",
                  ntrees = 123, max_depth = 6, stopping_tolerance = 0.00855107,
                  learn_rate = 0.08, distribution = "gaussian", sample_rate = 0.8,
                  col_sample_rate = 0.7, col_sample_rate_per_tree = 0.4)
h2o.saveModel(model3, "model3_gbm2.h2omodel")
model4 <- h2o.randomForest(x = c(4:17, 20:27), y = 3, training_frame = train,
                           nfolds = 5, validation_frame = valid, seed = 1,
                           keep_cross_validation_predictions = T,
                           score_tree_interval = 5, fold_assignment = "Modulo",
                           ntrees = 149, max_depth = 6,
                           stopping_tolerance = 0.00855107,
                           distribution = "gaussian", sample_rate = 0.8,
                           col_sample_rate = 0.7, col_sample_rate_per_tree = 0.4)
h2o.saveModel(model4, "model4_rf1.h2omodel")

h2o.performance(model1, valid)
h2o.performance(model2, valid)
h2o.performance(model3, valid)
h2o.performance(model4, valid)
# Each individual model has RMSE > $123,000, which is not acceptable

# Preparing data for stacking using glm - gaussian with alpha = lambda = 0
val_pred1 <- as.vector(h2o.predict(model1, valid))
val_pred2 <- as.vector(h2o.predict(model2, valid))
val_pred3 <- as.vector(h2o.predict(model3, valid))
val_pred4 <- as.vector(h2o.predict(model4, valid))
val_pred <- as.h2o(data.frame(pred1 = val_pred1, pred2 = val_pred2,
                              pred3 = val_pred3, pred4 = val_pred4,
                              price = as.vector(valid$price)))

# Performing stacking on validation data
model <- h2o.glm(x = 1:4, y = 5, training_frame = val_pred, alpha = 0, lambda = 0)
h2o.saveModel(model, "model_glm_stack.h2omodel")

# Preparing train set for stack prediction
train_pred1 <- as.vector(h2o.predict(model1, train))
train_pred2 <- as.vector(h2o.predict(model2, train))
train_pred3 <- as.vector(h2o.predict(model3, train))
train_pred4 <- as.vector(h2o.predict(model4, train))
train_pred <- as.h2o(data.frame(pred1 = train_pred1, pred2 = train_pred2,
                                pred3 = train_pred3, pred4 = train_pred4,
                                price = as.vector(train$price)))

# Checking performance: Pass
h2o.performance(model, val_pred)
# H2ORegressionMetrics: glm
# 
# MSE:  14245673772
# RMSE:  119355.2
# MAE:  69591.35
# RMSLE:  0.1778241
# Mean Residual Deviance :  14245673772
# R^2 :  0.8971434
# Null Deviance :8.013626e+14
# Null D.o.F. :5785
# Residual Deviance :8.242547e+13
# Residual D.o.F. :5781
# AIC :151707

h2o.performance(model, train_pred)
# H2ORegressionMetrics: glm
# 
# MSE:  6570075770
# RMSE:  81056
# MAE:  54123.24
# RMSLE:  0.1479952
# Mean Residual Deviance :  6570075770
# R^2 :  0.9507772
# Null Deviance :1.826083e+15
# Null D.o.F. :13675
# Residual Deviance :8.985236e+13
# Residual D.o.F. :13671
# AIC :347979.6

# Preparing test set for stack prediction and performance checking (Pass)
test_pred1 <- as.vector(h2o.predict(model1, test))
test_pred2 <- as.vector(h2o.predict(model2, test))
test_pred3 <- as.vector(h2o.predict(model3, test))
test_pred4 <- as.vector(h2o.predict(model4, test))
test_pred <- as.h2o(data.frame(pred1 = test_pred1, pred2 = test_pred2,
                               pred3 = test_pred3, pred4 = test_pred4,
                               price = as.vector(test$price)))
h2o.performance(model, test_pred)
# H2ORegressionMetrics: glm
# 
# MSE:  13199469757
# RMSE:  114888.9
# MAE:  69164.1
# RMSLE:  0.1775601
# Mean Residual Deviance :  13199469757
# R^2 :  0.9007037
# Null Deviance :2.85964e+14
# Null D.o.F. :2150
# Residual Deviance :2.839206e+13
# Residual D.o.F. :2146
# AIC :56241.98

h2o.shutdown()
# Retried after h2o.shutdown() and got simliar performance
