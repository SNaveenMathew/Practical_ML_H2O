library(h2o)
h2o.init()

# Importing and understanding the data set
in_data <- h2o.importFile("http://coursera.h2o.ai/cacao.882.csv")
summary(in_data)
# 0) Maker, Origin, Review Date, Cocoa Percent, Maker Location, Bean Type and Bean Origin are known at the time of Rating
# 1) Omitting NA values - removes 2 rows
# 2) REF looks categorical, but has 440 distinct values -> may lead to overfitting
# 3) Converting Review Date to factor
in_data <- h2o.na_omit(in_data)
in_data[, "Review Date"] <- h2o.asfactor(in_data[, "Review Date"])

# Splitting the data
parts <- h2o.splitFrame(in_data, c(0.7, 0.15), seed = 1)
train <- parts[[1]]
test <- parts[[2]]
valid <- parts[[3]]

# Independent and dependent variables
x <- c(1, 2, 4:6, 8, 9)
y <- 7

# Modeling
system.time(
  baseline_model <- h2o.deeplearning(x = x, y = y,
                                     training_frame = train,
                                     validation_frame = valid,
                                     epochs = 200, # Setting this to see when things start failing on Flow
                                     seed = 1))
# Flow shows overfitting at very low number of epochs. Still maintaining this as baseline
h2o.performance(baseline_model, train)
# MSE:  0.005301895
# RMSE:  0.07281411
# MAE:  0.053051
# RMSLE:  0.01798417
# Mean Residual Deviance :  0.005301895 -> Overfit (see test performance)

h2o.performance(baseline_model, valid)
# MSE:  0.2348637
# RMSE:  0.4846273
# MAE:  0.3721289
# RMSLE:  0.1227214
# Mean Residual Deviance :  0.2348637

h2o.performance(baseline_model, test)
# MSE:  0.2093796
# RMSE:  0.4575801
# MAE:  0.3612274
# RMSLE:  0.1158451
# Mean Residual Deviance :  0.2093796 -> Overfit!

# Made grid with best number of epochs = 2 and chose hyperparameters
system.time(best_model <- h2o.deeplearning(x = x, y = y, training_frame = train,
                                           validation_frame = valid, l1 = 1.0e-5,
                                           l2 = 1.0e-5, activation = "Tanh",
                                           epochs = 1, hidden = c(16, 8, 2),
                                           seed = 1))

h2o.performance(best_model, train)
# MSE:  0.2107307
# RMSE:  0.4590541
# MAE:  0.3732636
# RMSLE:  0.1158179
# Mean Residual Deviance :  0.2107307

h2o.performance(best_model, valid)
# MSE:  0.2370228
# RMSE:  0.4868498
# MAE:  0.3866118
# RMSLE:  0.1244031
# Mean Residual Deviance :  0.2370228

h2o.performance(best_model, test)
# MSE:  0.240354
# RMSE:  0.4902591
# MAE:  0.3901273
# RMSLE:  0.1266124
# Mean Residual Deviance :  0.240354
# Model performance isn't superior, but it's certainly not an overfit.

h2o.saveModel(object = baseline_model, path = "baseline_model")
h2o.saveModel(object = best_model, path = "best_model")

# Code results are more or less the same after rerunning
h2o.shutdown()
