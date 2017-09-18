#KNN function that takes a training set with a rightmost response column 
#takes a single row of a test set and returns the predicted factor level
#to be applied to each row of a test set
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

myKNN = function(train, testVal, k) {
  distances = data.frame(matrix(nrow=nrow(train), ncol = length(testVal)))
  for(i in 1:nrow(train)) {
    distances[i,] = (train[i, -length(train)] - testVal[1,])^2
  }
  squares = sqrt(apply(distances, 1, sum))
  trainComp = cbind(train, squares)
  trainComp = trainComp[order(squares),]
  trainComp = trainComp[1:k,]
  print(trainComp)
  return(Mode(trainComp$response))
}

