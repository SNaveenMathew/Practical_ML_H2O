source("initialize_iris.R")

auto_ml_models <- h2o.automl(1:4, 5, train)

auto_ml_models
summary(auto_ml_models)

auto_ml_models@leaderboard

preds <- h2o.predict(auto_ml_models@leader, test)

h2o.performance(auto_ml_models@leader, test)

h2o.shutdown()
