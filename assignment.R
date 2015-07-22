setwd("C:/Users/user/Dropbox/coursera/practical machine learning")
download.file(
    'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv', 
    destfile="training data.csv")

download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv', 
              destfile="testing data.csv")

training<-read.csv("training data.csv", sep=",", 
                   na.strings=c("", "NA", "NULL", "#DIV/0!"))

testing<-read.csv("testing data.csv", sep=",", 
                   na.strings=c("", "NA", "NULL", "#DIV/0!"))

str(training)
head(training)
dim(training)
dim(testing)

#checking if we have too many NA values for some variables; 
#then subsetting and dropping variables 
colSums(is.na(training))
training2<-training[, colSums(is.na(training))<19000]
dim(training2)

vector = c('X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 
           'cvtd_timestamp', 'new_window', 'num_window')
training3<-training2[, -which(names(training2) %in% vector)]

#loading caret package and performing some pre-analysis
library(caret)
#Checking if any of the left features have near zero variability
#training3<-training2[,-60]
nzv<-nearZeroVar(training3, saveMetrics=TRUE)
x<-sum(ifelse(nzv$zeroVar=="TRUE", 1, 0))
x
#[1] 0

#building a correlation matrix for the remaining vars, only can do it for numeric vars
corrMatrix<-cor(training3[sapply(training3, is.numeric)])
library(corrplot)
corrplot(corrMatrix, order = "FPC", method = "color",  tl.cex = 0.8, 
                  tl.col = rgb(0, 0, 0))
#remove variables with high correlation as they are not likely to predict well
removecor = findCorrelation(corrMatrix, cutoff = .80, verbose = TRUE)
training4=training3[-removecor]
dim(training4)

#splitting the obtained data set into training and testing
inTrain<-createDataPartition(training3$classe, p=0.7, list=FALSE)
train<-training4[inTrain ,]
test<-training4[-inTrain ,]
dim(train)
dim(test)

#Bagging model
library(randomForest)
set.seed(345)
fit1=randomForest(classe~., data=train, mtry=40, importance=TRUE)
#mtry=47 means all 47 predictors should be considered for each split of the tree
#i.e. bagging should be performed
#error rate is low, abour 1.43%
#predicting on the test set
yhat1=predict(fit1, newdata=test)
confusionMatrix(yhat1, test$classe)
varImpPlot(fit1, )

#Evaluate on the test data
tr_pred11=predict(fit1,test,type="class")
predMatrix = with(test,table(tr_pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix))


#random forest model, which by default has sqrt(p)when building a rf of classification trees
set.seed(3456)
fit2=randomForest(classe~., data=train, mtry=6, importance=TRUE)
yhat2=predict(fit2, newdata=test)
confusionMatrix(fit2, test$classe)
varImpPlot(fit2 , )

#Evaluate on test data and see better prediction from fit1
tr_pred=predict(fit2,test,type="class")
predMatrix = with(test,table(tr_pred,classe))
sum(diag(predMatrix))/sum(as.vector(predMatrix))




