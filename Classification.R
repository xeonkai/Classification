##### datasets #####
library(RCurl)
data(iris)
iris

UCI_data_URL <- getURL('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data')
names <- c('id_number', 'diagnosis', 'radius_mean', 
           'texture_mean', 'perimeter_mean', 'area_mean', 
           'smoothness_mean', 'compactness_mean', 
           'concavity_mean','concave_points_mean', 
           'symmetry_mean', 'fractal_dimension_mean',
           'radius_se', 'texture_se', 'perimeter_se', 
           'area_se', 'smoothness_se', 'compactness_se', 
           'concavity_se', 'concave_points_se', 
           'symmetry_se', 'fractal_dimension_se', 
           'radius_worst', 'texture_worst', 
           'perimeter_worst', 'area_worst', 
           'smoothness_worst', 'compactness_worst', 
           'concavity_worst', 'concave_points_worst', 
           'symmetry_worst', 'fractal_dimension_worst')
breast_cancer <- read.table(textConnection(UCI_data_URL), sep = ',', col.names = names)

breast_cancer$id_number <- NULL

##### libraries #####
library(randomForest)
library(MASS)
library(caret)
set.seed(100)

##### iris #####
training_samples <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
Train <- iris[training_samples, ]
Test <- iris[-training_samples, ]

# fitting models
rfModel <- randomForest(Species ~ ., data = Train, ntree = 2000)
qdaModel <- qda(Species ~ ., data = Train)

# QDA
qdaTrainPred <- predict(qdaModel, Train)
qdaTestPred <- predict(qdaModel, Test)

Train$QDAprob <- qdaTrainPred$posterior[, "setosa"]
Test$QDAprob <- qdaTestPred$posterior[, "setosa"]

# RF
rfTestPred <- predict(rfModel, Test, type = "prob")
Test$RFprob <- rfTestPred[, "setosa"]
Test$RFclass <- predict(rfModel, Test, type = "response")

# 
iristabs <- table(qdaTestPred$class, Test$Species)
sensitivity(iristabs, "virginica")
specificity(iristabs, c("setosa", "versicolor"))

posPredValue()

caret::confusionMatrix(data = qdaTestPred$class, reference = Test$Species)

# log loss


##### breast cancer #####
training_samples <- createDataPartition(breast_cancer$diagnosis, p = 0.7, list = FALSE)
Train <- breast_cancer[training_samples, ]
Test <- breast_cancer[-training_samples, ]

# fitting models
rfModel <- randomForest(diagnosis ~ ., data = Train, ntree = 2000)
qdaModel <- qda(diagnosis ~ ., data = Train)

# QDA
qdaTrainPred <- predict(qdaModel, Train)
qdaTestPred <- predict(qdaModel, Test)

Train$QDAprob <- qdaTrainPred$posterior[, "setosa"]
Test$QDAprob <- qdaTestPred$posterior[, "setosa"]

# RF
rfTestPred <- predict(rfModel, Test, type = "prob")
Test$RFprob <- rfTestPred[, "setosa"]
Test$RFclass <- predict(rfModel, Test, type = "response")
