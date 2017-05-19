# read in the data
x.train <- read.csv('data/dengue_features_train.csv', stringsAsFactors = T)
y.train<- read.csv('data/dengue_labels_train.csv', stringsAsFactors = T)
test.set <- read.csv('data/dengue_features_test.csv', stringsAsFactors = T)
# pad the test set with NA's so we can treat it the same way
test.set$total_cases=NA

# match up the lables with the actual data
x.train <- merge(x.train, y.train, by=c('city','year','weekofyear'))



# combine the train and test sets together to treat the factors in the same way
treating <- rbind(x.train, test.set)
#~~~~~~~~~~~~~~~~~~~~~~~~~~
# ENGINEER SOME VARIABLES
#~~~~~~~~~~~~~~~~~~~~~~~~~~
treating$month <- as.numeric( gsub("(^.*-)([0-9]+)(-.*$)","\\2", treating$week_start_date) )


#~~~~~~~~~~~~~
# fillin NA's
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# reorder the columns a little
idx <- grep('^city$|^year$|^weekofyear$|^week_start_date$|^month$', names(treating))
treating <- treating[,c(names(treating)[idx], names(treating)[-idx])]

# aggregate the non NA feature values on the city, year, and month and use this median value
#     as the NA values.
na.pred = aggregate(.~city+month, data=treating[-na.idx,], median)

# Fill in na's for each feature
for(feature.idx in 6:dim(treating)[2]){
  idx <- which(is.na(treating[,feature.idx]))
  if(length(idx)>0){
    # loop through all the different na cases and map in the median for that location and month
    for(i in 1:length(idx)){
      cat("feature: ", names(treating)[feature.idx], "\ti: ", i, "\n")
      treating[idx,][i,feature.idx] <- na.pred[which(  (na.pred[,'city'] == treating[idx,][i,'city']) & (na.pred[,'month'] == treating[idx,][i,'month']) ), feature.idx]
    }
  }
}

# other options for filling in NA's:
#   1) fit a model to the data the non NA data
#   2) fill in with aggregate data for a higher resolution. 
#       Eg: aggregate on cit, year, month  and use this data where we can. 
#       If there are any NA's still left, back up and aggregaat on city and moth

# separate the training and testing sets now that we've treated the values
ntrain <- dim(x.train)[1]
x.train <- treating[1:ntrain,]
test.set <- treating[(ntrain+1):(nrow(treating)),]










#~~~~~~~~~~~~~~~~~~~~~~~~~~
# TREATE THE VARIABLES - convert factors to numeric. necessary for some models
#~~~~~~~~~~~~~~~~~~~~~~~~~~
# ntrain <- dim(x.train)[1]
# features = names(treating)
# for (f in features) {
#   if (class(treating[[f]])=="factor") {
#     #cat("VARIABLE : ",f,"\n")
#     levels <- sort(unique(treating[[f]]))
#     treating[[f]] <- as.integer(factor(treating[[f]], levels=levels))
#   }
# }






# make some exploratory plots
#~~~~~~~~~~~~~~~~~~~~~~~~~~~
# par(mfrow = c(1, 2))
# for(i in 1:dim(x.train)[2]){
#   print(i)
#   if(class(x.train[,i])!='factor'){
#     plot(x.train[,i], main=names(x.train)[i])
#     hist(x.train[,i], main=names(x.train)[i])
#   }
# }



# # remove NA's in order to plot them  --- no
# idx = which(unlist(lapply(x.train, class)) != "factor")
# na.idx <- which(apply(x.train[,idx],1, function(y){return(any(is.na(y)))}))
# nona <- x.train[-na.idx,]
# tmp <- cor(nona[,idx])
# heatmap(tmp)








#~~~~~~~~~~~~~
# create training and testing data for CV
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




#~~~~~~~~~~~~~
# Model data GBM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
{
# library(gbm)
# library(caret)
# 
# fitControl <- trainControl(## 10-fold CV
#                             method = "repeatedcv",
#                             number = 5,
#                             ## repeated ten times
#                             repeats = 5)
# gbmGrid <- expand.grid(interaction.depth = (1:5) * 2, n.trees = (1:10)*50, shrinkage = .1, n.minobsinnode=1)
# set.seed(2)
# x.lab <- grep("total_cases", names(x.train) )
# 
# system.time( gbmFit <- train(x.train[,-x.lab], x.train[,x.lab], method = "gbm", metric="MAE", trControl = fitControl, verbose = T, bag.fraction = 0.5, tuneGrid = gbmGrid) )
# gbmFit
# plot(gbmFit)
# plot(gbmFit, metric = "RMSE")
# plot(gbmFit, metric = "Rsquared")
# plot(gbmFit, plotType = "level")
# 
# # predict using the test set !!! Also rounding the results
# test.set$total_cases <- round( predict(gbmFit, test.set) )
# 
# 
# # put the city name back into correct format
# test.set$city[which(test.set$city == '2')] = 'sj'
# test.set$city[which(test.set$city == '1')] = 'iq'
# 
# 
# out_file = 'submissions/gbm5x5_rmse20.868_r20.77.csv'
# write.csv(test.set[,c('city','year','weekofyear','total_cases')], file=out_file, quote=F, row.names=F)
}


#~~~~~~~~~~~~~
# Model data glm.nb
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Calculate MAE
mae <- function(actual, predicted){
  return(mean(abs(actual-predicted)) )
}







# Perform cross validation on a negative binomial GLM
runGLMnb <- function(x, cv1, cv2, model.formula){
  set.seed(12345)
  model.mae <- list()
  pb <- txtProgressBar(min=1, max=cv1, style=3)
  for(i in 1:cv1){
    
    # create test and train sample sets
    idx <- sample(1:dim(x)[1], dim(x)[1], replace=F)
    idx = as.numeric(cut(idx, cv2))
    
    for(cvbin in unique(idx)){
      idx.test <- which(idx %in% cvbin)
      
      # fit model
      gmod <- glm.nb(model.formula, data=x[-idx.test,])
      # gmod <- step(gmod)
      
      tmp = predict(gmod, x[idx.test,], type='response')
      
      model.mae[[paste(i,cvbin, sep="-")]] <- mae(x$total_cases[idx.test], tmp)
      
    }
    setTxtProgressBar(pb, i)
  }
  close(pb)
  return(data.frame(mean_mae=mean(unlist(model.mae)), median_mae=median(unlist(model.mae))))
}






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Negative Binomial GLM
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x <- x.train

model.formula <- as.formula("total_cases~.")
model.err2x2 <- runGLMnb(x, cv1=2, cv2=2, model.formula)
model.err5x5 <- runGLMnb(x, cv1=5, cv2=5, model.formula)
model.err10x10 <- runGLMnb(x, cv1=10, cv2=10, model.formula)

totalMAE <- data.frame(model="All_vars", rbind(model.err2x2, model.err5x5, model.err2x2), stringsAsFactors = F)



model.formula <- as.formula("total_cases ~ weekofyear + ndvi_se + ndvi_sw + reanalysis_avg_temp_k + reanalysis_dew_point_temp_k + reanalysis_specific_humidity_g_per_kg + station_avg_temp_c + station_max_temp_c")
model.err2x2 <- runGLMnb(x, cv1=2, cv2=2, model.formula)
model.err5x5 <- runGLMnb(x, cv1=5, cv2=5, model.formula)
model.err10x10 <- runGLMnb(x, cv1=10, cv2=10, model.formula)

totalMAE <- rbind( totalMAE, data.frame(model="low_sig", rbind(model.err2x2, model.err5x5, model.err2x2), stringsAsFactors = F) )

  
model.formula <- as.formula("total_cases ~ reanalysis_dew_point_temp_k + reanalysis_specific_humidity_g_per_kg + station_avg_temp_c + station_max_temp_c")
model.err2x2 <- runGLMnb(x, cv1=2, cv2=2, model.formula)
model.err5x5 <- runGLMnb(x, cv1=5, cv2=5, model.formula)
model.err10x10 <- runGLMnb(x, cv1=10, cv2=10, model.formula)

totalMAE <- rbind( totalMAE, data.frame(model="high_sig", rbind(model.err2x2, model.err5x5, model.err2x2), stringsAsFactors = F) )

# totalMAE.noSTEP <- totalMAE



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Adaboost
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
library(caret)

caret.mae <- function(dat, lev=NULL, model=NULL){
  return(c(MAE=mean(abs(dat$obs-dat$pred))) )
}



model.formula <- as.formula("total_cases~.")

fitControl <- trainControl(## X-fold CV
                            method = "repeatedcv",
                            number = 5,
                            ## repeated X times
                            repeats = 5,
                            summaryFunction = caret.mae,
                            verboseIter = TRUE
                            )
tunegrid <- expand.grid(.mtry=c(5:15))


set.seed(12345)
library(doMC)
registerDoMC(cores = 4)

x.lab <- grep("total_cases", names(x.train) )
# method ="rpart"
system.time( rfFit <- train(x.train[,-x.lab], x.train[,x.lab], method = "rf", metric="MAE", maximize=F, trControl = fitControl, verbose = T, bag.fraction = 0.5, tuneGrid = tunegrid, allowParallel=TRUE) )
rfFit
plot(rfFit)


test.set$total_cases <- NULL
tmp <- predict(rfFit$finalModel, test.set)




































