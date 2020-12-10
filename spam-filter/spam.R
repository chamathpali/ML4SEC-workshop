library(wordcloud)
library(tm)
require(tm)
library(caret)
library(e1071)

myData <- read.csv("spam_ham_dataset.csv") # read the dataset - Donwload from https://www.kaggle.com/venky73/spam-mails-dataset
myData<-myData[,c("label","text")] # select two columns
myData$text<-substring(myData$text,9) # removing "Subject:"

spamMsg<-subset(myData,label=="spam")
hamMsg<-subset(myData,label=="ham")

wordcloud(spamMsg$text,scale=c(4,.5),min.freq=5) # plot if the word apperas more than 5 times 
wordcloud(hamMsg$text,scale=c(4,.5),min.freq=5) # plot if the word apperas more than 5 times

myCorpus <- VCorpus(VectorSource(myData$text)) # create a corpus
myDTM <- DocumentTermMatrix(myCorpus, control = list( tolower=T, removeNumbers=T, removePunctuation=T, stopwords = T, stem=T ))

freqWords <- findFreqTerms(myDTM,5)
myDTM <- myDTM[,freqWords]

tr_index <- createDataPartition(myData$label, p=0.80, list=FALSE) # List of 80% of the rows
trainSet <- myData[tr_index,] # select 80% of the data for the trainSet
testSet <- myData[-tr_index,] # Select the remaining 20% of data for testSet

myDTMTrain <- myDTM[tr_index,]
myDTMTest <- myDTM[-tr_index,]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "T", "F")
}

myDTMTrainNew <- apply(myDTMTrain, MARGIN = 2,convert_counts)
myDTMTestNew <- apply(myDTMTest, MARGIN = 2, convert_counts)

#trainSet$label <- as.factor(trainSet$label) - ONLY MAC USERS

NBbasedSpamFilter <- naiveBayes(myDTMTrainNew, trainSet$label) # train the model

testPredictMsgLabel <- predict(NBbasedSpamFilter, myDTMTestNew) # predict labels for test cases

confusionMatrix(testPredictMsgLabel, testSet$label, positive = "spam") # Print confusion matrix