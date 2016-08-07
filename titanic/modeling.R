library(ggplot2)
library(caret)
library(pROC)
library(glmnet)

train.raw <- read.csv('train.csv', stringsAsFactors = F)
test.raw <- read.csv('test.csv', stringsAsFactors = F)


####################################################
###   BEGIN FUNCTIONS  ###
##########################

# Feature Engineering
#~~~~~~~~~~~~~~~~~~~~
engineerFeatures <- function(dat, response){
  # Look at titles of each passenger
  dat$title <- gsub("(^.*, )([A-Za-z]+)(\\. .*)","\\2",dat$Name)
  # now rename everything to age appropriate titles
#   titles <- data.frame( title=c("Mr","Mrs","Miss","Master","Don","Rev","Dr","Mme","Ms","Major","Lady","Sir","Mlle","Col","Capt","Dona"),
#                         title2=c("Mr","Mrs","Miss","Master","Sir","Sir","Sir","Lady","Ms","Mr","Lady","Sir","Lady","Mr","Sir","Lady"), stringsAsFactors=F )
#   dat <- merge(dat, titles, by="title")
#   dat$title <- NULL
#   names(dat)[grep("title2", names(dat))] <- "title"
  
  # Fill in NA's in age with the average age of those with the same title
  title.ages <- aggregate(Age~title, dat, mean, na.rm=T)
  dat <- merge(dat, title.ages, by="title", all.x=T)
  dat$Age.x[which(is.na(dat$Age.x))] = dat$Age.y[which(is.na(dat$Age.x))]
  dat$Age.y = NULL
  names(dat)[grep("Age.x",names(dat))] = "Age"
  
  # get cabin level
  dat$cabin_level <- gsub("(^[A-Za-z]{1})(.*$)", "\\1", dat$Cabin)
  
  
  
  # setting everything that needs to be a factor as a factor
  dat$Name <- as.factor(dat$Name)
  dat$Sex <- as.factor(dat$Sex)
  dat$Ticket <- as.factor(dat$Ticket)
  dat$Cabin <- as.factor(dat$Cabin)
  dat$Embarked <- as.factor(dat$Embarked)
  dat$title <- as.factor(dat$title)
  dat$cabin_level <- as.factor(dat$cabin_level)
  
  if(response==TRUE){
    dat <- dat[,c('Pclass', 'Age', 'Sex', 'title', "cabin_level",'Embarked', 'Survived')]
  }else{
    dat <- dat[,c('PassengerId','Pclass', 'Age', 'Sex', 'title', "cabin_level",'Embarked')]
  }
  
  titanicDummy <- dummyVars("~.",data=dat, fullRank=F)
  dat <- as.data.frame(predict(titanicDummy,dat))
  
  
  return(dat)
}


# Quick script to organize the predicted data into a submitable format
makeSubmission <- function(test_data, pred_data){
  submission <- data.frame(PassengerId = test_data$PassengerId, Survived = pred_data)
  submission$Survived = ifelse(submission$Survived=="died",0,1)
  return(submission)
}


####################################################
###   END FUNCTIONS  ###
##########################




# process data and create new features
train.raw <- engineerFeatures(dat=train.raw, response=T)
test.raw <- engineerFeatures(dat=test.raw, response=F)

# There may be differences between what's found in the training set vs the test set
missing_columns <- setdiff(names(train.raw), names(test.raw))
missing_columns <- missing_columns[-grep("Survived", missing_columns)]
test.raw[,missing_columns] <- NA


# split training data so we can have an in-house metric before submitting
idx <- sample(1:dim(train.raw)[1], floor(.75*dim(train.raw)[1]), replace=F)
train.set <- train.raw[idx,]
train.test <- train.raw[-idx,]

# split up x and y for some models
train.y <- as.factor(ifelse(train.set[,"Survived"]==1,"lived","died"))
train.x <- train.set[,-grep("Survived", names(train.set))]


######################################################
# ~~~~~~~~~~~~~~~ BEGIN MODELS ~~~~~~~~~~~~~~~~~~~~~~
######################################################



############
# GMB1
######
set.seed(1852)
# set gbm controls
ctrl1 <- trainControl(method='repeatedcv', 
                      number=3, 
                      preProcOptions = list(thresh = 0.95),
                      returnResamp='none', 
                      summaryFunction = twoClassSummary, 
                      classProbs = TRUE)

# train gbm model
gbm1 <- train(x=train.x, y=train.y, 
              method='gbm', 
              trControl=ctrl1,  
              metric = "ROC",
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
gbm1.pred <- predict(object=gbm1, test.raw[,-grep("PassengerId", names(test.raw))])

# write out a submission file
write.table(makeSubmission(test_data=test.raw, pred_data=gbm1.pred), "submissions/gbm6.csv", row.names=F, quote=F, sep=",")







###########################   <------------------ Winner so far
# using a parameter grid 
set.seed(1852)

# Play with parameters a bit
gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 3),
                        n.trees = (0:50)*10, 
                        shrinkage = seq(.0005, .05,.005),
                        n.minobsinnode = 10)

# train gbm model
system.time(gbm1.grid <- train(x=train.x, y=train.y, 
                                method='gbm', 
                                trControl=ctrl1, 
                                tuneGrid = gbmGrid,
                                metric = "ROC",
                                verbose=F,
                                preProc = c("center", "scale"))
)
# save(gbm1.grid, file="gbm.grid1.RData")

plot(gbm1.grid)
plot(varImp(gbm1.grid), top = 10)



# make a prediction with our inhouse test set
gbm1.pred <- predict(object=gbm1.grid, train.test, type='prob')
# get the area under the curve
auc <- roc(train.test$Survived, gbm1.pred[[2]])
print(auc$auc)  # <- Area under the curve: 0.8727


# predict the outcome of the test set
gbm1.pred <- predict(object=gbm1.grid, test.raw[,-grep("PassengerId", names(test.raw))])

# write out a submission file
write.table(makeSubmission(test_data=test.raw, pred_data=gbm1.pred), "submissions/gbm7.csv", row.names=F, quote=F, sep=",")





##################
# Making a bigger grid

# Play with parameters a bit
gbmGrid <-  expand.grid(interaction.depth = c(1, 2, 3),
                        n.trees = (0:50)*50, 
                        shrinkage = seq(.0005, .05,.0005),
                        n.minobsinnode = 10)

# train gbm model
system.time(gbm1.grid2 <- train(x=train.x, y=train.y, 
                                method='gbm', 
                                trControl=ctrl1, 
                                tuneGrid = gbmGrid,
                                metric = "ROC",
                                verbose=F,
                                preProc = c("center", "scale"))
)
# save(gbm1.grid2, file="gbm.grid1.RData")

plot(gbm1.grid2)
plot(varImp(gbm1.grid2), top = 10)

# make a prediction with our inhouse test set
gbm1.pred2 <- predict(object=gbm1.grid2, train.test, type='prob')
# get the area under the curve
auc <- roc(train.test$Survived, gbm1.pred2[[2]])
print(auc$auc)  # <- Area under the curve: 0.8796

# predict the outcome of the test set
gbm1.pred <- predict(object=gbm1.grid2, test.raw[,-grep("PassengerId", names(test.raw))])

# write out a submission file
write.table(makeSubmission(test_data=test.raw, pred_data=gbm1.pred), "C:/github/kaggle/titanic/submissions/gbm5.csv", row.names=F, quote=F, sep=",")


#

























































































# #### Ideas
# - Train a model to predict ages based on title, pchar, etc. Use for NA's
# - Use a "family" variable to take care of pchar and sibs
# - Married column based on Mr, Mrs
# - column that is 1 if a family member is listed to have survived

