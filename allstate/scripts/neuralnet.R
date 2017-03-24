library(neuralnet)
library(data.table)
library(Matrix)
library(xgboost)
library(Metrics)

ID = 'id'
TARGET = 'loss'
SEED = 0
SHIFT = 200

TRAIN_FILE = "~/github/kaggle/allstate/data/train.csv"  #"c:/github/kaggle/allstate/data/train.csv"
TEST_FILE = "~/github/kaggle/allstate/data/train.csv"  #"c:/github/kaggle/allstate/data/test.csv"
SUBMISSION_FILE = "~/github/kaggle/allstate/submit/chippys.txt"


train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

# y_train = log(train[,TARGET, with = FALSE] + SHIFT)[[TARGET]]
# 
# train[, c(ID, TARGET) := NULL]
train[, c(ID) := NULL]
test[, c(ID) := NULL]
# 
ntrain = nrow(train)
train_test = rbind(train, test)

features = names(train)

for (f in features) {
  if (class(train_test[[f]])=="character") {
    #cat("VARIABLE : ",f,"\n")
    levels <- sort(unique(train_test[[f]]))
    train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
  }
}

f <- as.formula(paste("loss ~", paste(n[!n %in% "loss"], collapse = " + ")))
train_test.mm = model.matrix(data=train_test, loss~.)


x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]




n <- names(x_train)
# f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))

nn <- neuralnet(f, data=x_train, hidden=c(1,1), threshold=0.1)
















