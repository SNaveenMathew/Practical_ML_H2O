source("Titanic/load_data.R")

automl_models <- h2o.automl(2:ncol(train), 1, train, seed = 1)
automl_models@leaderboard

h2o.performance(automl_models@leader, test)

h2o.predict(automl_models@leader, test)
