source("initialize_iris.R")

nb_model <- h2o.naiveBayes(1:4, 5, train)

h2o.performance(nb_model, train)

preds <- h2o.predict(nb_model, test)

h2o.performance(nb_model, test)

h2o.shutdown()
