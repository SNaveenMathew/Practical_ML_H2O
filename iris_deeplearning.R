source("initialize_iris.R")
summary(train)

dl_model <- h2o.deeplearning(1:4, 5, train, validation_frame = valid,
                             epochs = 40, stopping_rounds = 5,
                             stopping_metric = "logloss")
dl_model
summary(dl_model)

preds <- h2o.predict(dl_model, test)

h2o.performance(dl_model, test)

h2o.shutdown()
