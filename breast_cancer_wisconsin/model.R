library(ggplot2)
library(pROC)
library(caret)


x <- x.original <-  read.delim('data.txt', stringsAsFactors=F)




x$diagnosis = as.numeric(as.factor(x$diagnosis))-1

# Heatmap or correlation
x.cor = cor(x)
heatmap(x.cor)




# sort out training and testing sets
idx <- sample(1:dim(x)[1], ceiling(.1*dim(x)[1]))
x.test <- x[-idx,-2]
y.test <- x[-idx,2]
x.train <- x[idx,-2]
y.train <- x[idx,2]






#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# XGBOOST
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

require(xgboost)

set.seed(1234)
xgmod <- xgboost(data = data.matrix(x.train), label = y.train, max.depth = 2,
                 nrounds = 1000, objective = "reg:linear", print.every.n=5)

# figure out which features to keep
importance_matrix <- xgb.importance(names(x.train), model = xgmod)
importance_matrix$sum_gain <- cumsum(importance_matrix$Gain)
# xgb.plot.importance(importance_matrix)

best_features <- importance_matrix$Feature[which(importance_matrix$sum_gain<.99)]

#~~~~~~~~~~~~~~~~~~
# TAKE THE NEW FEATURE SET FOR A SPIN
set.seed(1234)
xgmod <- xgboost(data = data.matrix(x.train[,best_features]), label = y.train, max.depth = 12,
                 nrounds = 1000, objective = "reg:linear", print.every.n=5)

#~~~~~~~~~~~~~~~~~~

xg.pred <- predict(xgmod, data.matrix(x.test))
xg.auc <- roc(y.test, xg.pred)
plot(xg.auc)

xg.pred <- predict(object=xgmod, data.matrix(test.set))
xg.pred <- data.frame(id=test.set$id, loss=xg.pred)
write.table(xg.pred, '~/github/kaggle/allstate/submit/xgb_linear_n100_d12_top44.txt', quote=F, row.name=F, sep=',')


























