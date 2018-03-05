library(h2o)

h2o.init()

url <- "http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv"
iris <- h2o.importFile(url)

# Analysis is not reproducible. Revisit later if solution is found
parts <- h2o.splitFrame(iris, 0.8)
train <- parts[[1]]
test <- parts[[2]]

auto_ml_models <- h2o.automl(1:4, 5, train)

auto_ml_models
summary(auto_ml_models)

auto_ml_models@leaderboard

preds <- h2o.predict(auto_ml_models@leader, test)

h2o.performance(auto_ml_models@leader, test)

h2o.shutdown()
