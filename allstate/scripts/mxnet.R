require(mxnet)


train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

y_train = log(train[,TARGET, with = FALSE] + SHIFT)[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]

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

x_train = train_test[1:ntrain,]
x_test = train_test[(ntrain+1):nrow(train_test),]


# separate out the training and testing sets
idx <- sample(1:dim(x_train)[1], dim(x_train)[1]*.8)
x_train_train = data.matrix(x_train[-idx,])
x_train_test = data.matrix(x_train[idx,])
x_train_y = y_train[-idx]
x_test_y = y_train[idx]




# params for nn
params <- list(
  learning.rate = 0.0001,
  momentum = 0.9,
  batch.size = 100,
  wd = 0,
  num.round = 1000
)



# nn_model <- function(train_obs, test_obs, params) {
  inp <- mx.symbol.Variable('data')
  l1 <- mx.symbol.FullyConnected(inp, name = "l1", num.hidden = 400)
  a1 <- mx.symbol.Activation(l1, name = "a1", act_type = 'relu')
  d1 <- mx.symbol.Dropout(a1, name = 'd1', p = 0.4)
  l2 <- mx.symbol.FullyConnected(d1, name = "l2", num.hidden = 200)
  a2 <- mx.symbol.Activation(l2, name = "a2", act_type = 'relu')
  d2 <- mx.symbol.Dropout(a2, name = 'd2', p = 0.2)
  l3 <- mx.symbol.FullyConnected(d2, name = "l3", num.hidden = 1)
  outp <- mx.symbol.MAERegressionOutput(l3, name = "outp")
  
  m <- mx.model.FeedForward.create(outp, 
                                   X = as.array(t(x_train_train)), 
                                   y = as.array(x_train_y),
                                   eval.data =
                                     list(data = as.array(t(x_train_test)),
                                          label = as.array(x_test_y)),
                                   array.layout = 'colmajor',
                                   eval.metric=mx.metric.mae,
                                   learning.rate = params$learning.rate,
                                   momentum = params$momentum,
                                   wd = params$wd,
                                   array.batch.size = params$batch.size,
                                   num.round = params$num.round)
  

  # Check out the MAE of the training sets
  pred_train <- predict(m, as.array(t(x_train_train)), array.layout = 'colmajor')
  MAE_train <- mean(abs(pred_train - x_train_y))
  MAE_train
  
  pred <- predict(m, as.array(t(x_train_test)), array.layout = 'colmajor')  
  MAE_test <- mean(abs(pred - x_test_y))
  MAE_test
  cat("MAE training:", MAE_train, ",\tMAE testing :", MAE_test, "\n")
  
  # return(list(model = m, MAE_test = MAE_test, MAE_train = MAE_train))
# }
  
  
  # use model to predic with and write out submission results
  pred <- predict(m, as.array(t(data.matrix(x_test))), array.layout = 'colmajor')
  submission = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
  submission$loss = exp(pred) - SHIFT
  write.csv(submission,'submit/mxnet_1000init3l.txt',row.names = FALSE)
  
  
  
  
  
  
  
  
