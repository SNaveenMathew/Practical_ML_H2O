source("initialize_iris.R")
summary(train)

dl_model <- h2o.deeplearning(1:4, 5, train)
dl_model
summary(dl_model)

preds <- h2o.predict(dl_model, test)

h2o.performance(dl_model, test)

h2o.shutdown()
