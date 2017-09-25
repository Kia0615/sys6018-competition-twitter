# Load in the library
library(XML)
library(tm)
library(class)

# Read in the data
train <- as.data.frame(read.csv("train.csv"))
trainsentiment <- as.data.frame(train["sentiment"])
traintext <- as.data.frame(train["text"])
test <- as.data.frame(read.csv("test.csv"))
testtext <- as.data.frame(test["text"])

# Define the function to process the text
docRead = function(document.data.frame, sparsity) {
  document.data.frame = as.data.frame(document.data.frame, stringsAsFactors = FALSE)
  news = VCorpus(DataframeSource(document.data.frame))
  news.tfidf = DocumentTermMatrix(news, control = list(weighting = weightTfIdf))
  
  news.clean = tm_map(news, stripWhitespace)                          # remove extra whitespace
  news.clean = tm_map(news.clean, removeNumbers)                      # remove numbers
  news.clean = tm_map(news.clean, removePunctuation)                  # remove punctuation
  news.clean = tm_map(news.clean, content_transformer(tolower))       # ignore case
  news.clean = tm_map(news.clean, removeWords, stopwords("english"))  # remove stop words
  news.clean = tm_map(news.clean, stemDocument)
  
  news.clean.tfidf = DocumentTermMatrix(news.clean, control = list(weighting = weightTfIdf))
  clean = as.matrix(news.clean.tfidf)
  
  tfidf.sparse = removeSparseTerms(news.clean.tfidf, sparsity)
  cleansparse = as.matrix(tfidf.sparse)
  return(cleansparse)
}

# Define the function to calculate Euclidean distance
Eudist <- function(x1,x2){
  d=sum((x1-x2)^2)
  return (sqrt(d))
}

# Define the function that returns the mode
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# Define the k-nn function that takes one test data point
myKNN_1 <-  function(testVal,train,response, k) {
  index_y <- which(colnames(train)==response)
  trainx <- train[,-index_y]
  distances <- apply(trainx,1,Eudist,testVal)
  trainComp <- cbind(train, distances)
  trainComp <- trainComp[order(distances),]
  kset <- trainComp[1:k,]
  return(Mode(kset[,index_y]))
  
}

# Define the k-nn function that takes the whole test dataset
myKNN <- function(test,train,response, k){
  apply(test,1,FUN=myKNN_1,train=train,response=response,k=k)
}

# Clean data with .99 sparsity
trainvar <- as.data.frame(docRead(traintext,0.99))
traindata <- cbind(trainvar,trainsentiment)
testvar <- as.data.frame(docRead(testtext,0.99))
trainsentiment <- t(trainsentiment)

# Cross-validate to check which k-value yields the best result
results <- knn.cv(trainvar, trainsentiment, k = 1)
table(results,trainsentiment)
(1+25+421+41+3)/981 # 0.5005097

results <- knn.cv(trainvar, trainsentiment, k = 2)
table(results,trainsentiment)
(1+24+435+31+4)/981 #0.5045872

results <- knn.cv(trainvar, trainsentiment, k = 3)
table(results,trainsentiment)
(0+24+478+38+3)/981 #0.5535168

results <- knn.cv(trainvar, trainsentiment, k = 5)
table(results,trainsentiment)
(0+18+521+26+2)/981 #0.5779817

results <- knn.cv(trainvar, trainsentiment, k = 7)
table(results,trainsentiment)
(0+8+551+23+1)/981 #0.5942915

results <- knn.cv(trainvar, trainsentiment, k = 9)
table(results,trainsentiment)
(0+5+566+15+0)/981 #0.5973496

results <- knn.cv(trainvar, trainsentiment, k = 11)
table(results,trainsentiment)
(0+3+578+17+0)/981 #0.6095821

results <- knn.cv(trainvar, trainsentiment, k = 15)
table(results,trainsentiment)
(0+4+590+5+0)/981 #0.6106014

results <- knn.cv(trainvar, trainsentiment, k = 20)
table(results,trainsentiment)
(0+1+600+2+0)/981 #0.6146789

results <- knn.cv(trainvar, trainsentiment, k = 30)
table(results,trainsentiment)
(1+603)/981 #0.6156983


# Predict the test set sentiment valuest with k=10 & export the data
mypreds_k10 <- as.data.frame(myKNN(testvar,traindata,"sentiment",10))
mypreds_k10 <- cbind(test$id,mypreds_k10)
write.table(mypreds_k10, file = "k10.csv", row.names=F, col.names=c("id","sentiment"), sep=",")

# Predict the test set sentiment valuest with k=20 & export the data
mypreds_k20 <- as.data.frame(myKNN(testvar,traindata,"sentiment",20))
mypreds_k20 <- cbind(test$id,mypreds_k20)
write.table(mypreds_k20, file = "k20.csv", row.names=F, col.names=c("id","sentiment"), sep=",")
