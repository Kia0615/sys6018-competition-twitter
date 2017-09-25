library(tidyverse)
library(XML)
library(tm)

#function to take a document and get the weight matrix 
docRead = function(document.data.frame, sparsity) {
  
  document.data.frame = as.data.frame(document.data.frame, stringsAsFactors = FALSE)
  #create VCorpus
  news = VCorpus(DataframeSource(document.data.frame))
  #create weight matrix
  news.tfidf = DocumentTermMatrix(news, control = list(weighting = weightSMART))
  
  #clean data
  news.clean = tm_map(news, stripWhitespace)                          # remove extra whitespace
  news.clean = tm_map(news.clean, removeNumbers)                      # remove numbers
  news.clean = tm_map(news.clean, removePunctuation)                  # remove punctuation
  news.clean = tm_map(news.clean, content_transformer(tolower))       # ignore case
  news.clean = tm_map(news.clean, removeWords, stopwords("english"))  # remove stop words
  news.clean = tm_map(news.clean, stemDocument)
  
  #re-create weight matrix
  news.clean.tfidf = DocumentTermMatrix(news.clean, control = list(weighting = weightSMART))
  clean = as.matrix(news.clean.tfidf)
  
  #remove sparse terms based on sparsity argument
  tfidf.sparse = removeSparseTerms(news.clean.tfidf, sparsity)
  cleansparse = as.matrix(tfidf.sparse)
  return(cleansparse)
}

#read in train and test sets and get matrices
train = read_csv('train.csv')
train99 = as.data.frame(docRead(train, .99))

test = read_csv('test.csv')
test99 = as.data.frame(docRead(test, .99))

#only get variables that are in both
trainCols99 = names(train99)
testCols99 = names(test99)
common99 = intersect(trainCols99, testCols99)

train99 = as.data.frame(train99[ , (names(train99) %in% common99)])
test99 = as.data.frame(test99[ , (names(test99) %in% common99)])

#add sentiment to the training weight matrix
train99 = cbind(train99, train$sentiment)
colnames(train99)[colnames(train99) == 'train$sentiment'] <- 'sentiment'

#create models
model1 = lm(sentiment ~., data = train99)

model2 = step(model1)

model4 = lm(sentiment ~ dont + googl + need + thing + want + wait + less + insur + cant + 
              will + hit + come, data = train99)

model3 = lm(sentiment ~ polym(dont,googl,need,thing,want,wait,less,insur,cant,
           will,hit,come, raw=TRUE, degree=3), data = train99)


#reformat polynomial model to be used with cv.lm
mf = as.data.frame(model.matrix(sentiment ~ polym(dont,googl,need,thing,want,wait,less,insur,cant,
                                                  will,hit,come, raw=TRUE, degree=3), data = train99))
mf$sentiment = train99$sentiment
mf$'(Intercept)' = NULL
names(mf) <- make.names(names(mf))
g.poly = lm(sentiment ~ ., data = mf)
cv.lm(mf, g.poly, m = 5)$delta[1]

#perform LOOCV to evaluate
#CV was also performed with different weighting schemes used for the matrix
#as well as different variable selections for model4
cv.lm(train99, model1, m=nrow(train99))$delta[1]
cv.lm(train99, model2, m=nrow(train99))$delta[1]
cv.lm(train99, model3, m=nrow(train99))$delta[1]
cv.lm(train99, model4, m=nrow(train99))$delta[1]

#make predictions
probs = predict(model3, test99)
probs = lapply(probs, round, 0)
probs = as.data.frame(unlist(probs))
predictions = cbind(test$id, probs)
predictions[predictions$`unlist(probs)`>5, 2] = 5
predictions[predictions$`unlist(probs)`<1, 2] = 1
write.table(predictions, file = "lmattempt6.csv", row.names=F, col.names=c("id", "sentiment"), sep=",")






