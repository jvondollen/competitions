library(ggplot2)
library(caret)
library(pROC)
library(glmnet)
library(data.table)



dat<- data.frame(fread("c:/github/kaggle/allstate/data/train.csv"), stringsAsFactors=F)



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# adjust features

dat.class <- sapply(dat, class)

idx <- which(dat.class == "character")
for(i in idx){
  dat[,i] <- as.factor(dat[,i])  
}




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# keep a sample of the original training set since we have actual answers for this. 
idx <- sample(1:dim(dat)[1], floor(dim(dat)[1]*.25), replace=FALSE)

train.set <- dat[idx,]
train.test <- dat[-idx,]

train.x <- train.set[,-grep("loss",names(train.set))]
train.y <- train.set[,grep("loss",names(train.set))]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~






############
# GMB1
######
set.seed(1852)
# set gbm controls
ctrl1 <- trainControl(method='repeatedcv', 
                      number=3, 
                      preProcOptions = list(thresh = 0.95),
                      returnResamp='none', 
                      summaryFunction = twoClassSummary)

# train gbm model
gbm1 <- train(x=train.x, y=train.y, 
              method='gbm', 
              trControl=ctrl1,  
              metric = "RMSE",
              preProc = c("center", "scale"))

summary(gbm1)
gbm1

# make a prediction with our inhouse test set
gbm1.pred <- predict(object=gbm1, train.test, type='prob')
# get the area under the curve
auc <- roc(train.test$Survived, gbm1.pred[[2]])
print(auc$auc)  # <-  0.835
#                     0.8756 with consolidated titles


# predict the outcome of the test set
gbm1.pred <- predict(object=gbm1, test.set[,-grep("PassengerId", names(test.set))])












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

#



































require(xgboost)

model <- xgboost(data = train$data, label = train$label,
                 nrounds = 2, objective = "binary:logistic")
