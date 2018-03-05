library(h2o)

h2o.init()
url <- "http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv"
iris <- h2o.importFile(url)

# Analysis is not reproducible. Revisit later if solution is found
parts <- h2o.splitFrame(iris, 0.8)
train <- parts[[1]]
test <- parts[[2]]

summary(train)

dl_model <- h2o.deeplearning(1:4, 5, train)
dl_model
summary(dl_model)

preds <- h2o.predict(dl_model, test)

h2o.performance(dl_model, test)

h2o.shutdown()
