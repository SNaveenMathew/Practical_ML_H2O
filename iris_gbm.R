source("initialize_iris.R")

gbm_model <- h2o.gbm(1:4, 5, train)
gbm_model
summary(gbm_model)

preds <- h2o.predict(gbm_model, test)

h2o.performance(gbm_model, test)

h2o.shutdown()
