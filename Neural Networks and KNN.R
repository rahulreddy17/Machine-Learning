
###### NOTE: BOTH THE CODE AND REPORT ANALYSIS FOLLOW THE SAME ORDER, BRIEF COMMENTS ARE WRITTEN IN THE CODE TO UNDERSTAND WHATS BEING DONE ############

load.libraries <- c("devtools","keras","tensorflow","class", "readxl")

install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs)
sapply(load.libraries, require, character = TRUE)

install_keras()
install_tensorflow()

############# DATA SET 1: ONLINE NEWS POPULARITY ##################

############ ARTIFICIAL NEURAL NETWORKS ##############

##Data loading
news<-read.csv("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_3/data.csv")
summary(news)

###Data Pre processing
news$shares <- as.numeric(ifelse(news$shares < median(news$shares),0,1))

norm <- function(x){
  (x - min(x))/(max(x)-min(x))
}

news <- cbind(data.frame(lapply(news[,-59], norm)),shares = news$shares)

news <- as.matrix(news)
dimnames(news) <- NULL

set.seed(123)
ind <- sample(2, nrow(news), replace=TRUE, prob=c(0.67, 0.33))

##split the news data
news.training <- news[ind==1, 1:58]
news.test <- news[ind==2, 1:58]

# Split the class attribute
news.trainingtarget <- news[ind==1, 59]
news.testtarget <- news[ind==2, 59]

### Training a basic model and cross validation ############

# Initialize a sequential model
model <- keras_model_sequential()

# Add layers to the model
model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(58)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile the model
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

#Fit the model
history <- model %>% fit(
  news.training,
  news.trainingtarget, 
  epochs = 200, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history)

# Predict the classes for the test data
classes <- model %>% predict_classes(news.test, batch_size = 128)

# Confusion matrix
table(news.testtarget, classes)

# Evaluate on test data and labels
score <- model %>% evaluate(news.test, news.testtarget, batch_size = 128)

# Print the score
print(score$acc)

###############

###Experimentation with layers and units

###########
h <- c(10,15,20,25,30,35,40)
j <- c(5,10,15,20,25,30,35)
layers <- data.frame(cbind(h,j))
null <- data.frame(acc = NULL)
for(i in 1:7){
  model <- keras_model_sequential()
  # Add layers to the model
  model %>% 
    layer_dense(units = layers[i,]$h, activation = 'relu', input_shape = c(58)) %>% 
    layer_dense(units = layers[i,]$j, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')
  
  # Compile the model
  model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
  )
  
  #Fit the model
  history <- model %>% fit(
    news.training, 
    news.trainingtarget, 
    epochs = 30, 
    batch_size = 5, 
    validation_split = 0.2
  )
  
  
  # Predict the classes for the test data
  classes <- model %>% predict_classes(news.test, batch_size = 128)
  
  # Confusion matrix
  table(news.testtarget, classes)
  score <- model %>% evaluate(news.test, news.testtarget, batch_size = 128)
  acc <- score$acc
  null <- rbind(null,acc)
}
null
names(null) <- c("ACC")
final<-as.data.frame(cbind(layers$h,layers$j,null$ACC))
names(final)<- c("layer1","layer2","test_accuracy")
final
plot(final$test_accuracy)

###########

##### Experimentation with activation functions

######

#### ReLU activation function
model1 <- keras_model_sequential()
model1 %>% 
  layer_dense(units = 15, activation = 'relu', input_shape = c(58)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history1 <- model1 %>% fit(
  news.training,
  news.trainingtarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history1)
classes1 <- model1 %>% predict_classes(news.test, batch_size = 128)
table(news.testtarget, classes1)
score1 <- model1 %>% evaluate(news.test, news.testtarget, batch_size = 128)
print(score1$acc)

##### tanh activation function
model2 <- keras_model_sequential()
model2 %>% 
  layer_dense(units = 15, activation = 'tanh', input_shape = c(58)) %>% 
  layer_dense(units = 10, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history2 <- model2 %>% fit(
  news.training,
  news.trainingtarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history2)
classes2 <- model2 %>% predict_classes(news.test, batch_size = 128)
table(news.testtarget, classes2)
score2 <- model2 %>% evaluate(news.test, news.testtarget, batch_size = 128)
print(score2$acc)

##### sigmoid activation function
model3 <- keras_model_sequential()
model3 %>% 
  layer_dense(units = 15, activation = 'sigmoid', input_shape = c(58)) %>% 
  layer_dense(units = 10, activation = 'sigmoid') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model3 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history3 <- model3 %>% fit(
  news.training,
  news.trainingtarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history3)
classes3 <- model3 %>% predict_classes(news.test, batch_size = 128)
table(news.testtarget, classes3)
score3 <- model3 %>% evaluate(news.test, news.testtarget, batch_size = 128)
print(score3$acc)

act_fct <- as.data.frame(cbind(score1$acc,score2$acc,score3$acc))
names(act_fct)<- c("acc_ReLU","acc_tanh","acc_sigmoid")
act_fct

###########

##### Experimentation with optimization parameters

######
###SGD Optimizer
model4 <- keras_model_sequential()
model4 %>% 
  layer_dense(units = 15, activation = 'relu', input_shape = c(58)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
# Define an optimizer
sgd <- optimizer_sgd(lr = 0.01)

model4 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'sgd',
  metrics = 'accuracy'
)

history4 <- model4 %>% fit(
  news.training,
  news.trainingtarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history4)
classes4 <- model4 %>% predict_classes(news.test, batch_size = 128)
table(news.testtarget, classes4)
score4 <- model4 %>% evaluate(news.test, news.testtarget, batch_size = 128)
print(score4$acc)

acc_opt <- as.data.frame(cbind(score1$acc,score4$acc))
names(acc_opt) <- c("ADAM_acc","SGD_acc")
acc_opt

### Final model
model_final <- keras_model_sequential()
model_final %>% 
  layer_dense(units = 15, activation = 'relu', input_shape = c(58)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model_final %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history_final <- model_final %>% fit(
  news.training,
  news.trainingtarget, 
  epochs = 50, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history_final)
classes_final <- model_final %>% predict_classes(news.test, batch_size = 128)
table(news.testtarget, classes_final)
score_final <- model_final %>% evaluate(news.test, news.testtarget, batch_size = 128)
print(score_final$acc)


########### KNN on Dataset 1 ##################

news_KNN<-read.csv("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_3/data.csv")
set.seed(123)

###Data Pre processing

news_KNN$shares <- as.numeric(ifelse(news_KNN$shares < median(news_KNN$shares),0,1))

norm <- function(x){
  (x - min(x))/(max(x)-min(x))
}

news_norm <- cbind(data.frame(lapply(news_KNN[,-59], norm)),shares = news_KNN$shares)

ind <- sample(2, nrow(news_norm), replace=TRUE, prob=c(0.67, 0.33))

##split the news data
news_KNN.training <- news_norm[ind==1, 1:58]
news_KNN.test <- news_norm[ind==2, 1:58]

# Split the class attribute
news_KNN.trainingtarget <- news_norm[ind==1, 59]
news_KNN.testtarget <- news_norm[ind==2, 59]

## Running the basic model
prediction <- knn(train = news_KNN.training, test = news_KNN.test, cl= news_KNN.trainingtarget,k = 3)
CF<-table(prediction,news_KNN.testtarget)
CF
KNN_accuracy <- sum(diag(CF))/sum(CF[])
KNN_accuracy

## Fine tuning the model to decide the best K
a<-seq(3,60,by=3)
k <- 1:20
null <- data.frame(accuracy = NULL)
for(x in a[k]){
  prediction <- knn(train = news_KNN.training, test = news_KNN.test, cl= news_KNN.trainingtarget, k = x)
  CF<-table(prediction,news_KNN.testtarget)
  accuracy <- sum(diag(CF))/sum(CF[])
  finaltable <- as.data.frame(cbind(x,accuracy))
  null <- rbind(null,finaltable)
}
null

plot(null$x, null$accuracy, type = 'b')
##K=33 would be ideal

###Overall final model
prediction <- knn(train = news_KNN.training, test = news_KNN.test, cl= news_KNN.trainingtarget,k = 33)
CF<-table(prediction,news_KNN.testtarget)
CF
KNN_accuracy <- sum(diag(CF))/sum(CF[])
KNN_accuracy

############# DATA SET 2: ORGANICS PRODUCT DATA ##################

############ ARTIFICIAL NEURAL NETWORKS ##############

#importing the data and removed the non predictive variables
organics <- read_excel("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_3/organics.xlsx")
organics<-as.data.frame((organics))
organics$TargetAmt<-NULL
organics$ID<-NULL #since ID is just an index variable
organics$DemCluster<-NULL
lapply(organics, class)

##One hot encoding the variables
org_num <- as.matrix(organics[,-c(3,4,5,6,7)])

En_DemCluster<-stats::model.matrix(~DemClusterGroup, organics)[,-1]
En_DemGender<-stats::model.matrix(~DemGender, organics)[,-1]
En_DemReg<-stats::model.matrix(~DemReg, organics)[,-1]
En_DemTVReg<-stats::model.matrix(~DemTVReg, organics)[,-1]
En_PromClass<-stats::model.matrix(~PromClass, organics)[,-1]

organics_final <- cbind.data.frame(org_num,En_DemCluster,En_DemGender,En_DemReg,En_DemTVReg,En_PromClass)

###Data Normaliation
norm <- function(x){
  (x - min(x))/(max(x)-min(x))
}

organics_norm <- cbind(data.frame(lapply(organics_final[,-5], norm)),TargetBuy = organics_final$TargetBuy)

data_2 <- as.matrix(organics_norm)
dimnames(data_2) <- NULL

set.seed(123)
xx <- sample(2, nrow(data_2), replace=TRUE, prob=c(0.67, 0.33))

##split the news data
org_train <- data_2[xx==1, 1:30]
org_test <- data_2[xx==2, 1:30]

# Split the class attribute
org.traintarget <- data_2[xx==1, 31]
org.testtarget <- data_2[xx==2, 31]


##### Training a basic model and cross validating it ############

## Initialize a sequential model
Org_model <- keras_model_sequential()

# Add layers to the model
Org_model %>% 
  layer_dense(units = 10, activation = 'relu', input_shape = c(30)) %>% 
  layer_dense(units = 5, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile the model
Org_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

#Fit the model
Org_history <- Org_model %>% fit(
  org_train,
  org.traintarget, 
  epochs = 200, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(Org_history)

# Predict the classes for the test data
Org_classes <- Org_model %>% predict_classes(org_test, batch_size = 128)

# Confusion matrix
table(org.testtarget, Org_classes)

# Evaluate on test data and labels
Org_score <- Org_model %>% evaluate(org_test, org.testtarget, batch_size = 128)

# Print the score
print(Org_score$acc)

###############

###Experimentation with layers and units

###########

h <- c(10,15,20,25,30,35,40)
j <- c(5,10,15,20,25,30,35)
layers <- data.frame(cbind(h,j))
null <- data.frame(acc = NULL)
for(i in 1:7){
  
  Org_model <- keras_model_sequential()
  Org_model %>% 
    layer_dense(units = layers[i,]$h, activation = 'relu', input_shape = c(30)) %>% 
    layer_dense(units = layers[i,]$j, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')
  
  Org_model %>% compile(
    loss = 'binary_crossentropy',
    optimizer = 'adam',
    metrics = 'accuracy'
  )
  
  Org_history <- Org_model %>% fit(
    org_train,
    org.traintarget, 
    epochs = 30, 
    batch_size = 5, 
    validation_split = 0.2
  )
  
  plot(Org_history)
  
  # Predict the classes for the test data
  Org_classes <- Org_model %>% predict_classes(org_test, batch_size = 128)
  table(org.testtarget, Org_classes)
  Org_score <- Org_model %>% evaluate(org_test, org.testtarget, batch_size = 128)
  acc <- Org_score$acc
  null <- rbind(null,acc)
}
null
names(null) <- c("ACC")
final<-as.data.frame(cbind(layers$h,layers$j,null$ACC))
names(final)<- c("layer1","layer2","test_accuracy")
final
plot(final$test_accuracy)

###########

##### Experimentation with activation functions

######

#### ReLU activation function
Org_model1 <- keras_model_sequential()
Org_model1 %>% 
  layer_dense(units = 15, activation = 'relu', input_shape = c(30)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
Org_model1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

Org_history1 <- Org_model1 %>% fit(
  org_train,
  org.traintarget, 
  epochs = 50, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(Org_history1)

Org_classes1 <- Org_model1 %>% predict_classes(org_test, batch_size = 128)
table(org.testtarget, Org_classes1)
Org_score1 <- Org_model1 %>% evaluate(org_test, org.testtarget, batch_size = 128)
print(Org_score1$acc)

#### tanh activation function
Org_model2 <- keras_model_sequential()
Org_model2 %>% 
  layer_dense(units = 15, activation = 'tanh', input_shape = c(30)) %>% 
  layer_dense(units = 10, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'sigmoid')
Org_model2 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

Org_history2 <- Org_model2 %>% fit(
  org_train,
  org.traintarget, 
  epochs = 50, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(Org_history2)

Org_classes2 <- Org_model2 %>% predict_classes(org_test, batch_size = 128)
table(org.testtarget, Org_classes2)
Org_score2 <- Org_model2 %>% evaluate(org_test, org.testtarget, batch_size = 128)
print(Org_score2$acc)

#### Sigmoid activation function
Org_model3 <- keras_model_sequential()
Org_model3 %>% 
  layer_dense(units = 15, activation = 'sigmoid', input_shape = c(30)) %>% 
  layer_dense(units = 10, activation = 'sigmoid') %>%
  layer_dense(units = 1, activation = 'sigmoid')
Org_model3 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

Org_history3 <- Org_model3 %>% fit(
  org_train,
  org.traintarget, 
  epochs = 50, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(Org_history3)

Org_classes3 <- Org_model3 %>% predict_classes(org_test, batch_size = 128)
table(org.testtarget, Org_classes3)
Org_score3 <- Org_model3 %>% evaluate(org_test, org.testtarget, batch_size = 128)
print(Org_score3$acc)

act_fct <- as.data.frame(cbind(Org_score1$acc,Org_score2$acc,Org_score3$acc))
names(act_fct)<- c("acc_ReLU","acc_tanh","acc_sigmoid")
act_fct

###########

##### Experimentation with optimization parameters

######
###SGD Optimizer
Org_model4 <- keras_model_sequential()
Org_model4 %>% 
  layer_dense(units = 15, activation = 'tanh', input_shape = c(30)) %>% 
  layer_dense(units = 10, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'sigmoid')
# Define an optimizer
sgd <- optimizer_sgd(lr = 0.01)

Org_model4 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'sgd',
  metrics = 'accuracy'
)

Org_history4 <- Org_model4 %>% fit(
  org_train,
  org.traintarget, 
  epochs = 50, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(Org_history4)

Org_classes4 <- Org_model4 %>% predict_classes(org_test, batch_size = 128)
table(org.testtarget, Org_classes4)
Org_score4 <- Org_model4 %>% evaluate(org_test, org.testtarget, batch_size = 128)
print(Org_score4$acc)

acc_opt <- as.data.frame(cbind(Org_score2$acc,Org_score4$acc))
names(acc_opt) <- c("ADAM_acc","SGD_acc")
acc_opt

## Final Model
Org_model_f <- keras_model_sequential()
Org_model_f %>% 
  layer_dense(units = 15, activation = 'tanh', input_shape = c(30)) %>% 
  layer_dense(units = 10, activation = 'tanh') %>%
  layer_dense(units = 1, activation = 'sigmoid')
Org_model_f %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

Org_history_f <- Org_model_f %>% fit(
  org_train,
  org.traintarget, 
  epochs = 50, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(Org_history_f)

Org_classes_f <- Org_model_f %>% predict_classes(org_test, batch_size = 128)
table(org.testtarget, Org_classes_f)
Org_score_final <- Org_model_f %>% evaluate(org_test, org.testtarget, batch_size = 128)
print(Org_score_final$acc)

############### KNN on Dataset 2 Organics data############

organics <- read_excel("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_3/organics.xlsx")
organics<-as.data.frame((organics))
organics$TargetAmt<-NULL
organics$ID<-NULL #since ID is just an index variable
organics$DemCluster<-NULL
lapply(organics, class)

##One hot encoding the variables
org_num <- as.matrix(organics[,-c(3,4,5,6,7)])

En_DemCluster<-stats::model.matrix(~DemClusterGroup, organics)[,-1]
En_DemGender<-stats::model.matrix(~DemGender, organics)[,-1]
En_DemReg<-stats::model.matrix(~DemReg, organics)[,-1]
En_DemTVReg<-stats::model.matrix(~DemTVReg, organics)[,-1]
En_PromClass<-stats::model.matrix(~PromClass, organics)[,-1]

organics_final <- cbind.data.frame(org_num,En_DemCluster,En_DemGender,En_DemReg,En_DemTVReg,En_PromClass)

###Data Normaliation
norm <- function(x){
  (x - min(x))/(max(x)-min(x))
}

organics_norm <- cbind(data.frame(lapply(organics_final[,-5], norm)),TargetBuy = organics_final$TargetBuy)

ind <- sample(2, nrow(organics_norm), replace=TRUE, prob=c(0.67, 0.33))

##split the news data
organics_KNN.training <- organics_norm[ind==1, 1:30]
organics_KNN.test <- organics_norm[ind==2, 1:30]

# Split the class attribute
organics_KNN.trainingtarget <- organics_norm[ind==1, 31]
organics_KNN.testtarget <- organics_norm[ind==2, 31]

set.seed(100)
org_prediction <- knn(train = organics_KNN.training, test = organics_KNN.test, cl= organics_KNN.trainingtarget,k = 3)
Org_CF<-table(org_prediction,organics_KNN.testtarget)
Org_CF
Org_KNN_accuracy <- sum(diag(Org_CF))/sum(Org_CF[])
Org_KNN_accuracy


a<-seq(3,60,by=3)
k <- 1:20
Org_null <- data.frame(Org_KNN_accuracy = NULL)
for(x in a[k]){
  org_prediction <- knn(train = organics_KNN.training, test = organics_KNN.test, cl= organics_KNN.trainingtarget,k = x)
  Org_CF<-table(org_prediction,organics_KNN.testtarget)
  Org_KNN_accuracy <- sum(diag(Org_CF))/sum(Org_CF[])
  Org_finaltable <- as.data.frame(cbind(x,Org_KNN_accuracy))
  Org_null <- rbind(Org_null,Org_finaltable)
}
Org_null

plot(Org_null$x, Org_null$Org_KNN_accuracy, type = 'b')
##K=9 would be ideal

##Final Model
org_prediction <- knn(train = organics_KNN.training, test = organics_KNN.test, cl= organics_KNN.trainingtarget,k = 9)
Org_CF<-table(org_prediction,organics_KNN.testtarget)
Org_CF
Org_KNN_accuracy <- sum(diag(Org_CF))/sum(Org_CF[])
Org_KNN_accuracy
