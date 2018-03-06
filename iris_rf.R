source("initialize_iris.R")

rf_model <- h2o.randomForest(1:4, 5, train)
rf_model
summary(rf_model)

preds <- h2o.predict(rf_model, test)

h2o.performance(rf_model, test)

h2o.shutdown()
