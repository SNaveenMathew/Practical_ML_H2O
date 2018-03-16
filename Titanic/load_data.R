library(h2o)
library(mice)
library(readr)

process_data <- function(data, type = "train") {
  if(type == "train") {
    data$Survived[data$Survived==1] <- "Survived"
    data$Survived[data$Survived!="Survived"] <- "Died"
    data$PassengerId <- NULL
  }
  data$Name <- strsplit(data$Name, ", ")
  data$Name2 <- sapply(data$Name, function(u) u[1])
  data$Name1 <- sapply(data$Name, function(u) u[2])
  data$Name <- NULL
  data$Title <- strsplit(data$Name1, "\\. ")
  data$Title <- sapply(data$Title, function(u) u[1])
  data$Title[data$Title %in%
               c("Don", "Major", "Capt", "Jonkheer", "Rev", "Col")] <- "Mr"
  data$Title[data$Title %in% c("Countess", "Mme")] <- "Mrs"
  data$Title[data$Title %in% c("Mlle", "Ms")] <- "Miss"
  data$Title[data$Title == "Dr" & data$Sex == "Male"] <- "Mr"
  data$Title[data$Title == "Dr" & data$Sex == "Female"] <- "Mrs"
  data$Cabin <- sapply(strsplit(data$Cabin, "[0-9]"), function(u) u[1])
  data$Cabin[is.na(data$Cabin)] <- "Unknown"
  data$FamilySize <- data$SibSp + data$Parch
  data$FareRatio <- data$Fare / (data$FamilySize + 1)
  data$Ticket <- data$Name1 <- data$Name2 <- NULL
  if("Survived" %in% colnames(data))
    data$Survived <- as.factor(data$Survived)
  data$Pclass <- as.factor(data$Pclass)
  data$Sex <- as.factor(data$Sex)
  data$Cabin <- as.factor(data$Cabin)
  data$Embarked <- as.factor(data$Embarked)
  missing <- sapply(data, function(col) sum(is.na(col)))
  columns <- colnames(data)[missing <= 10]
  columns <- setdiff(columns, c("PassengerId", "Survived"))
  data$Total <- data$SibSp + data$Parch + 1
  mice_model <- mice(data[,columns], m = 500, seed = 1)
  lis <- mice_model$imp
  non_null <- sapply(lis, function(u) !is.null(u))
  lis <- lis[non_null]
  lis <- sapply(lis, function(df)
    if(nrow(df) > 1){
      return(apply(X = df, MARGIN = 1, FUN = rowMedian))
    } else{
      return(rowMedian(df))
    })
  if(ncol(lis) > 0) {
    if(ncol(lis) > 1) {
      for(col in colnames(lis))
        data[is.na(data[, col]), col] <- lis[, col]
    } else {
      data[is.na(data[, col]), col] <- lis[, col]
    }
  }
  return(data)
}

rowMedian <- function(row) {
  tab <- table(as.character(row))
  tab <- tab[order(tab, decreasing = T)]
  return(names(tab)[1])
}

h2o.init()
train <- read_csv("Titanic/train.csv")
train <- process_data(train)

train <- h2o::as.h2o(train)
parts <- h2o.splitFrame(
  train, 0.8, destination_frames = c("train", "test"), seed = 1)
train <- parts[[1]]
test <- parts[[2]]
