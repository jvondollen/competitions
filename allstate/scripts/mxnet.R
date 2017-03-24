require(mxnet)



# params for nn
params_nn <- list(
  learning.rate = 0.0001,
  momentum = 0.9,
  batch.size = 100,
  wd = 0,
  num.round = 100
)



nn_model <- function(train_obs, test_obs, params) {
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
                                   X = as.array(t(data[train_obs, -y])), 
                                   y = as.array(data[train_obs, y]),
                                   eval.data =
                                     list(data = as.array(t(data[test_obs, -y])),
                                          label = as.array(data[test_obs, y])),
                                   array.layout = 'colmajor',
                                   eval.metric=mx.metric.mae,
                                   learning.rate = params$learning.rate,
                                   momentum = params$momentum,
                                   wd = params$wd,
                                   array.batch.size = params$batch.size,
                                   num.round = params$num.round)
  
  pred <- predict(m, as.array(t(data[test_obs, -y])), array.layout = 'colmajor')
  pred_train <- predict(m, as.array(t(data[train_obs, -y])), array.layout = 'colmajor')
  
  MAE_test <- mean(abs(pred - data[test_obs, y]))
  MAE_train <- mean(abs(pred_train - data[train_obs, y]))
  
  return(list(model = m, MAE_test = MAE_test, MAE_train = MAE_train))
}
