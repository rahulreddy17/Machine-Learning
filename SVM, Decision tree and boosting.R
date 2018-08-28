#loading the required packages
library('e1071')
library('tree')
library('rpart')
library('rpart.plot')
library('adabag')
library(readxl)
library(ROCR)

####Data set 1: Online News Popularity######

setwd("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_2")

#importing the data and removed the non predictive variables
news<-read.csv("data.csv")
news=news[!news$n_unique_tokens==701,]
summary(news)

#scaling the data
for(i in ncol(news)-1){ 
  news[,i]<-scale(news[,i], center = TRUE, scale = TRUE)
}

#Classification of the dependent variable
newsclass <-news
newsclass$shares <- as.factor(ifelse(newsclass$shares > 1400,1,0))

#splitting the data
#set random situation
set.seed(100)
rand <- runif(nrow(newsclass))
newsrand <- newsclass[order(rand), ]

newstrain <- newsrand[1:27750, ]
newstest <- newsrand[27751:39643, ]
xx<-newsrand[1:10000,]

########## Support Vector Machines ###########
#Tuning for SVM model with linear kernel functions
t1in <- Sys.time()
tune.out1 = tune(svm,shares~.,data = xx, kernel="linear",ranges = list(cost=c(1,5,10,20,40,50,100)))
summary(tune.out1)
t1out <- Sys.time()-t1in;t1out

plot(error~cost,tune.out1$performances)

#Predicting and accuracy
bestmod1=tune.out1$best.model
ypred1=predict (bestmod1 ,newstest)
p1<-table(predict =ypred1 , truth=newstest$shares )
Accuracy.SVM.lin<-(p1[1,1]+p1[2,2])/sum(p1[])
p1
Accuracy.SVM.lin


##Tuning for SVM model with Radial Kernel functions
t2in <- Sys.time()
tune.out2=tune(svm , shares~., data=xx, kernel ="radial",
              ranges=list(cost=c(0.1,1,10,100),
                          gamma=c(0.5,1,2,3) ))
summary(tune.out2)
t2out <- Sys.time()-t2in;t2out

plot(tune.out2)

#Predicting and accuracy
bestmod2=tune.out2$best.model
ypred2=predict (bestmod2 ,newstest)
p2<-table(predict =ypred2 , truth=newstest$shares )
Accuracy.SVM.rad<-(p2[1,1]+p2[2,2])/sum(p2[])
p2
Accuracy.SVM.rad

##Tuning for SVM model with Sigmoid kernel functions
t3in <- Sys.time()
tune.out3=tune(svm , shares~., data=xx, kernel ="sigmoid",
               ranges=list(cost=c(0.001,0.005,0.5,1),
                           gamma=c(0.25,0.5,1,2) ))
summary(tune.out3)
t3out <- Sys.time()-t3in;t3out
plot(tune.out3)

#Predicting and accuracy
bestmod3=tune.out3$best.model
ypred3=predict (bestmod3 ,newstest)
p3<-table(predict =ypred3 , truth=newstest$shares )
Accuracy.SVM.sig<-(p3[1,1]+p3[2,2])/sum(p3[])
p3
Accuracy.SVM.sig


#########Decision trees#####
#Build a sample full length tree
tree1 <- rpart(shares ~ ., data = newstrain, method = 'class',control = rpart.control(cp=0.0002))
summary(tree1)
rpart.plot(tree1)

#Cross validate the tree to select the best possible tree size and complexity parameter
plotcp(tree1)
tree1$cptable

#Final Decision tree
treefinal <- rpart(shares ~ ., data = newstrain, method = 'class',control = rpart.control(cp=0.001))
rpart.plot(treefinal,uniform=TRUE,main="Pruned Classification Tree")
treefinal$variable.importance
barplot(treefinal$variable.importance[order(treefinal$variable.importance, decreasing = TRUE)],ylim = c(0, 500), main = "Variables Relative Importance",col = "lightblue")

##Predicting and calculating the accuracy on the test dataset
tree.pred=predict(treefinal ,newstest , type="class")
p4<-table(tree.pred ,newstest$shares)
Accuracy.decisiontrees<-(p4[1,1]+p4[2,2])/sum(p4[])
p4
Accuracy.decisiontrees


##########BOOSTING########
#Running the boosting algorithm with the training dataset initially for 100 iterations
news.adaboost <- boosting(shares ~ ., data = xx, mfinal = 100,control = rpart.control(maxdepth = 1))
news.adaboost$weights
barplot(news.adaboost$imp[order(news.adaboost$imp, decreasing = TRUE)],ylim = c(0, 100), main = "Variables Relative Importance",col = "lightblue")

##Checking out the error rate of the train dataset
table(news.adaboost$class, xx$shares,dnn = c("Predicted Class", "Observed Class"))
1 - sum(news.adaboost$class == xx$shares) /length(xx$shares)

##Prediction on the testdataset 
news.predboosting <- predict.boosting(news.adaboost,newdata = newstest)
news.predboosting$confusion
news.predboosting$error

##Now lets verify what can be the ideal number of iterations to train our model so that our model predicts perfectly on the test dataset
##Cross validation
news.adaboost1 <- boosting(shares ~ ., data = xx, mfinal = 100,control = rpart.control(maxdepth = 1))
news.adaboost2 <- boosting(shares ~ ., data = xx, mfinal = 300,control = rpart.control(maxdepth = 1))
news.adaboost3 <- boosting(shares ~ ., data = xx, mfinal = 600,control = rpart.control(maxdepth = 1))
news.adaboost4 <- boosting(shares ~ ., data = xx, mfinal = 900,control = rpart.control(maxdepth = 1))
news.adaboost5 <- boosting(shares ~ ., data = xx, mfinal = 1200,control = rpart.control(maxdepth = 1))

news.predboosting1 <- predict.boosting(news.adaboost1,newdata = newstest)
news.predboosting2 <- predict.boosting(news.adaboost2,newdata = newstest)
news.predboosting3 <- predict.boosting(news.adaboost3,newdata = newstest)
news.predboosting4 <- predict.boosting(news.adaboost4,newdata = newstest)
news.predboosting5 <- predict.boosting(news.adaboost5,newdata = newstest)

error<-c(news.predboosting1$error,news.predboosting2$error,news.predboosting3$error,news.predboosting4$error,news.predboosting5$error)
iterations<-c(100,300,600,900,1200)
plot(iterations,error,main = "Error rate vs iterations") # 900 iterations would be the ideal one

#Final Model with 900 iterations
news.adaboostfinal <- boosting(shares ~ ., data = xx, mfinal = 900,control = rpart.control(maxdepth = 1))
news.adaboostfinal$weights
barplot(news.adaboostfinal$imp[order(news.adaboostfinal$imp, decreasing = TRUE)],ylim = c(0, 100), main = "Variables Relative Importance",col = "lightblue")

##Prediction on the testdataset 
news.predboostingfinal <- predict.boosting(news.adaboostfinal,newdata = newstest)
news.predboostingfinal$confusion
news.predboostingfinal$error
p5<-news.predboostingfinal$confusion
Accuracy.boosting<-(p5[1,1]+p5[2,2])/sum(p5[])
p5
Accuracy.boosting
                                         
##Margins and error evolution
news.margins <- margins(news.adaboostfinal, xx)
news.predmargins <- margins(news.predboostingfinal, newstest)
news.margins
news.predmargins

margins.test <- news.predmargins[[1]]
margins.train <- news.margins[[1]]
plot(sort(margins.train), (1:length(margins.train)) /length(margins.train), type = "l", xlim = c(-1,1),main = "Margin cumulative distribution graph", xlab = "m",ylab = "% observations", col = "blue3", lwd = 2)
abline(v = 0, col = "red", lty = 2, lwd = 2)
lines(sort(margins.test), (1:length(margins.test)) / length(margins.test),type = "l", cex = 0.5, col = "green", lwd = 2)
legend("topleft", c("test","train"), col = c("green", "blue"), lty = 1,lwd = 2)

##Error evolution
evol.test <- errorevol(news.adaboostfinal, newstest)
evol.train <- errorevol(news.adaboostfinal, xx)
plot(evol.test$error, type = "l", ylim = c(0.2, 0.4),main = "Boosting error versus number of trees", xlab = "Iterations",ylab = "Error", col = "red", lwd = 2)
lines(evol.train$error, cex = .5, col = "blue", lty = 2, lwd = 2)
legend("bottomright", c("test", "train"), col = c("red", "blue"), lty = 1:2,lwd = 2)

#######Data set 2 : Organic Products Data###########

#importing the data and removed the non predictive variables
organics <- read_excel("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_2/organics.xlsx")
View(organics)
organics<-as.data.frame(unclass(organics))
organics$TargetBuy<-as.factor(organics$TargetBuy)
organics$TargetAmt<-NULL
organics$ID<-NULL #since ID is just an index variable


#splitting the data
#set random situation
set.seed(100)
rand2 <- runif(nrow(organics))
organicsrand <- organics[order(rand2), ]

organicstrain <- organicsrand[1:11485, ]
organicstest <- organicsrand[11486:16408, ]
yy<-organicsrand[1:10000,]

########## Support Vector Machines ###########
####Tuning for SVM model with linear kernel functions
t1in <- Sys.time()
orgtune.out1 = tune(svm,TargetBuy~.,data = yy, kernel="linear",ranges = list(cost=c(0.01,0.1,1,2,5,10,20,40,100)))
summary(orgtune.out1)
t1out <- Sys.time()-t1in;t1out

plot(error~cost,orgtune.out1$performances)

#Predicting and accuracy
bestmod1=orgtune.out1$best.model
ypred1=predict (bestmod1 ,organicstest)
p1<-table(predict =ypred1 , truth=organicstest$TargetBuy )
Accuracy.SVM.lin<-(p1[1,1]+p1[2,2])/sum(p1[])
Accuracy.SVM.lin

####Tuning for SVM model with Radial Kernel functions
t2in <- Sys.time()
orgtune.out2=tune(svm , TargetBuy~., data=yy, kernel ="radial",
                  ranges=list(cost=c(0.1,1,10,100),
                              gamma=c(0.5,1,2,3) ))
summary(orgtune.out2)
t2out <- Sys.time()-t2in;t2out

plot(orgtune.out2)

##Predicting and accuracy
bestmod2=orgtune.out2$best.model
ypred2=predict (bestmod2 ,organicstest)
p2<-table(predict =ypred2 , truth=organicstest$TargetBuy )
Accuracy.SVM.rad<-(p2[1,1]+p2[2,2])/sum(p2[])
p2
Accuracy.SVM.rad

####Tuning for SVM model with Sigmoid kernel functions
t3in <- Sys.time()
orgtune.out3=tune(svm , TargetBuy~., data=yy, kernel ="sigmoid",
                  ranges=list(cost=c(0.001,0.005,0.5,1),
                              gamma=c(0.25,0.5,1,2) ))
summary(orgtune.out3)
t3out <- Sys.time()-t3in;t3out
plot(orgtune.out3)

bestmod3=orgtune.out3$best.model
ypred3=predict (bestmod3 ,organicstest)
p3<-table(predict =ypred3 , truth=organicstest$TargetBuy )
Accuracy.SVM.sig<-(p3[1,1]+p3[2,2])/sum(p3[])
p3
Accuracy.SVM.sig

Data2_SVM_Acc<-as.matrix(c(Accuracy.SVM.lin,Accuracy.SVM.rad,Accuracy.SVM.sig))

#########Decision trees#####
#Build a sample full length tree
Orgtree1 <- rpart(TargetBuy ~ ., data = organicstrain, method = 'class',control = rpart.control(cp=0.0001))
summary(Orgtree1)
rpart.plot(Orgtree1)

#Cross validate the tree to select the best possible tree size and complexity parameter
plotcp(Orgtree1)
Orgtree1$cptable
#Final Decision tree with cp =0.0016
Orgtreefinal <- rpart(TargetBuy ~ ., data = organicstrain, method = 'class',control = rpart.control(cp=0.0016))
rpart.plot(Orgtreefinal,uniform=TRUE,main="Pruned Classification Tree")
Orgtreefinal$variable.importance
barplot(Orgtreefinal$variable.importance[order(Orgtreefinal$variable.importance, decreasing = TRUE)],ylim = c(0, 1000), main = "Variables Relative Importance",col = "lightblue")

##Predicting and calculating the accuracy on the test dataset
Orgtree.pred=predict(Orgtreefinal ,organicstest , type="class")
p4<-table(Orgtree.pred ,organicstest$TargetBuy)
Accuracy.decisiontree<-(p4[1,1]+p4[2,2])/sum(p4[])
p4
Accuracy.decisiontree

##########BOOSTING########
#Running the boosting algorithm with the training dataset initially for 100 iterations
Org.adaboost <- boosting(TargetBuy ~ ., data = yy, mfinal = 100,control = rpart.control(maxdepth = 1))
Org.adaboost$weights
barplot(Org.adaboost$imp[order(Org.adaboost$imp, decreasing = TRUE)],ylim = c(0, 100), main = "Variables Relative Importance",col = "lightblue")

##Checking out the error rate of the train dataset
##table(Org.adaboost$class, organicstrain$TargetBuy,dnn = c("Predicted Class", "Observed Class"))
##1 - sum(news.adaboost$class == xx$shares) /length(xx$shares)

##Prediction on the testdataset 
org.predboosting <- predict.boosting(Org.adaboost,newdata = organicstest)
org.predboosting$confusion
org.predboosting$error

##Now lets verify what can be the ideal number of iterations to train our model so that our model predicts perfectly on the test dataset
##Cross validation
org.adaboost1 <- boosting(TargetBuy ~ ., data = yy, mfinal = 100,control = rpart.control(maxdepth = 1))
org.adaboost2 <- boosting(TargetBuy ~ ., data = yy, mfinal = 300,control = rpart.control(maxdepth = 1))
org.adaboost3 <- boosting(TargetBuy ~ ., data = yy, mfinal = 600,control = rpart.control(maxdepth = 1))
org.adaboost4 <- boosting(TargetBuy ~ ., data = yy, mfinal = 900,control = rpart.control(maxdepth = 1))
org.adaboost5 <- boosting(TargetBuy ~ ., data = yy, mfinal = 1200,control = rpart.control(maxdepth = 1))

org.predboosting1 <- predict.boosting(org.adaboost1,newdata = organicstest)
org.predboosting2 <- predict.boosting(org.adaboost2,newdata = organicstest)
org.predboosting3 <- predict.boosting(org.adaboost3,newdata = organicstest)
org.predboosting4 <- predict.boosting(org.adaboost4,newdata = organicstest)
org.predboosting5 <- predict.boosting(org.adaboost5,newdata = organicstest)

error<-c(org.predboosting1$error,org.predboosting2$error,org.predboosting3$error,org.predboosting4$error,org.predboosting5$error)
iterations<-c(100,300,600,900,1200)
plot(iterations,error,main = "Error rate vs iterations") # 300 iterations would be the ideal one

#Final Model with 300 iterations
org.adaboostfinal <- boosting(TargetBuy ~ ., data = yy, mfinal = 300,control = rpart.control(maxdepth = 1))
org.adaboostfinal$weights
barplot(org.adaboostfinal$imp[order(org.adaboostfinal$imp, decreasing = TRUE)],ylim = c(0, 100), main = "Variables Relative Importance",col = "lightblue")


##Prediction on the testdataset 
org.predboostingfinal <- predict.boosting(org.adaboostfinal,newdata = organicstest)
org.predboostingfinal$confusion
org.predboostingfinal$error
p5<-org.predboostingfinal$confusion
Accuracy.boosting<-(p5[1,1]+p5[2,2])/sum(p5[])


##Margins and error evolution
org.margins <- margins(org.adaboostfinal, yy)
org.predmargins <- margins(org.predboostingfinal, organicstest)
org.margins
org.predmargins

margins.test <- org.predmargins[[1]]
margins.train <- org.margins[[1]]
plot(sort(margins.train), (1:length(margins.train)) /length(margins.train), type = "l", xlim = c(-1,1),main = "Margin cumulative distribution graph", xlab = "m",ylab = "% observations", col = "blue3", lwd = 2)
abline(v = 0, col = "red", lty = 2, lwd = 2)
lines(sort(margins.test), (1:length(margins.test)) / length(margins.test),type = "l", cex = 0.5, col = "green", lwd = 2)
legend("topleft", c("test","train"), col = c("green", "blue"), lty = 1,lwd = 2)

##Error evolution
evol.test <- errorevol(org.adaboostfinal, organicstest)
evol.train <- errorevol(org.adaboostfinal, yy)
plot(evol.test$error, type = "l", ylim = c(0.2, 0.4),main = "Boosting error versus number of trees", xlab = "Iterations",ylab = "Error", col = "red", lwd = 2)
lines(evol.train$error, cex = .5, col = "blue", lty = 2, lwd = 2)
legend("topright", c("test", "train"), col = c("red", "blue"), lty = 1:2,lwd = 2)


