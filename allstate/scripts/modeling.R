library(ggplot2)
library(caret)
library(pROC)
library(glmnet)
library(data.table)






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# adjust features and engineering

# convert the character columns to factors
convert2factors <- function(dat){
  dat.class <- sapply(dat, class)
  
  idx <- which(dat.class == "character")
  for(i in idx){
    dat[,i] <- as.factor(dat[,i])  
  }
  return(dat)
}



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Read in training data
dat <- data.frame(fread("c:/github/kaggle/allstate/data/train.csv"), stringsAsFactors=F)
# convert character columns to factors
dat <- convert2factors(dat)

# read in test set
dat.test <- data.frame(fread("c:/github/kaggle/allstate/data/test.csv"), stringsAsFactors=F)
# convert character columns to factors
test.set <- convert2factors(dat.test)




# keep a sample of the original training set since we have actual answers for this. 
idx <- sample(1:dim(dat)[1], floor(dim(dat)[1]*.75), replace=FALSE)

train.set <- dat[idx,]
train.test <- dat[-idx,]

train.x <- train.set[,-grep("loss",names(train.set))]
train.y <- train.set[,grep("loss",names(train.set))]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






# ############
# # GMB1
# ######
# set.seed(1852)
# # set gbm controls
# ctrl1 <- trainControl(method='repeatedcv', 
#                       number=3, 
#                       preProcOptions = list(thresh = 0.95),
#                       returnResamp='none', 
#                       summaryFunction = twoClassSummary)
# 
# # train gbm model
# gbm1 <- train(x=train.x, y=train.y, 
#               method='gbm', 
#               trControl=ctrl1,  
#               metric = "RMSE",
#               preProc = c("center", "scale"))
# 
# summary(gbm1)
# gbm1
# 
# # make a prediction with our inhouse test set
# gbm1.pred <- predict(object=gbm1, train.test, type='prob')
# # get the area under the curve
# auc <- roc(train.test$Survived, gbm1.pred[[2]])
# print(auc$auc)  # <-  0.835
# #                     0.8756 with consolidated titles
# 
# 
# # predict the outcome of the test set
# gbm1.pred <- predict(object=gbm1, test.set[,-grep("PassengerId", names(test.set))])










# FIRST GBM

fitControl <- trainControl(method = "cv", number = 5)


tune_Grid <-  expand.grid(interaction.depth = 2,
                          n.trees = 500,
                          shrinkage = 0.1,
                          n.minobsinnode = 10)


set.seed(825)
gbm.fit <- train(x=train.x, y=train.y, method = "gbm",
               trControl = fitControl, verbose = FALSE,
               tuneGrid = tune_Grid)


summary(gbm.fit)
attributes(gbm.fit)

#

# make a prediction with our inhouse test set
gbm.pred <- predict(object=gbm.fit, train.test)
# get the area under the curve
gbm.auc <- roc(train.test$loss, gbm.pred)
print(gbm.auc$auc)  # <-  0.835
#                     0.8756 with consolidated titles


# predict the outcome of the test set
# gbm1.pred <- predict(object=gbm.pred, test.set[,-grep("PassengerId", names(test.set))])
gbm.pred <- predict(object=gbm.fit, test.set)
gbm.pred <- data.frame(id=test.set$id, loss=gbm.pred, stringsAsFactors=F)

write.table(gbm.pred, 'c:/github/kaggle/allstate/submit/gbm1.5_75percent.txt', quote=F, row.names=F, sep=',')

marginal plot per predictor

#




# XGBoost model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


require(xgboost)

set.seed(1234)
xgmod <- xgboost(data = data.matrix(train.x), label = train.y,
                 nrounds = 1000,
                 objective = "reg:linear", print.every.n=5)

xg.pred <- predict(xgmod, data.matrix(train.test))
xg.auc <- roc(train.test$loss, xg.pred)
plot(xg.auc)

xg.pred <- predict(object=xgmod, data.matrix(test.set))
xg.pred <- data.frame(id=test.set$id, loss=xg.pred)
write.table(xg.pred, 'c:/github/kaggle/allstate/submit/xgb_linear_n1000.txt', quote=F, row.name=F, sep=',')



























require(xgboost)

model <- xgboost(data = train$data, label = train$label,
                 nrounds = 2, objective = "binary:logistic")
