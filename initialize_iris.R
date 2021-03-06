library(h2o)

h2o.init()
url <- "http://h2o-public-test-data.s3.amazonaws.com/smalldata/iris/iris_wheader.csv"
iris <- h2o.importFile(url)

# Analysis is not reproducible. Revisit later if solution is found
parts <- h2o.splitFrame(iris, c(0.7, 0.15), seed = 1)
train <- parts[[1]]
test <- parts[[2]]
valid <- parts[[3]]
