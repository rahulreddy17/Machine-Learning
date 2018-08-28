############# DATA SET 1: ONLINE NEWS POPULARITY ##################

###### NOTE: BOTH THE CODE AND REPORT ANALYSIS FOLLOW THE SAME ORDER, BRIEF COMMENTS ARE WRITTEN IN THE CODE TO UNDERSTAND WHATS BEING DONE ############

load.libraries <- c("tidyverse","cluster","factoextra", "ClusterR", "randomForest","ggthemes","rsvd","gridExtra", "mclust", "fastICA","moments","devtools","keras","tensorflow","class", "readxl")

install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs)
sapply(load.libraries, require, character = TRUE)

install_keras()
install_tensorflow()

### TASK 1 : Clustering Alogrithms

### K- MEANS

##Data loading
news<-read.csv("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_4/data.csv")
summary(news)

###Data Pre processing
news$shares <- as.numeric(ifelse(news$shares < median(news$shares),0,1))

norm <- function(x){
  (x - min(x))/(max(x)-min(x))
}

news <- cbind(data.frame(lapply(news[,-59], norm)),shares = news$shares)
news_Kmeans<- news[,-59]

##Basic clustering with 2 centres
k2 <- kmeans(news_Kmeans, centers = 2, nstart = 1)
str(k2)
k2
fviz_cluster(k2, data = news_Kmeans)

##Function for elbow method
wss <- function(k) {
  kmeans(news_Kmeans, k, nstart = 2 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 10
k.values <- 1:10

# extract wss for 2-10 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 10, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

### from the plot we can say that 8 would be the ideal number of clusters
## computing kmeans clustering with k = 8
final <- kmeans(news_Kmeans, 8, nstart = 10)
print(final)

fviz_cluster(final, data = news_Kmeans)

##### EXPECTATION MAXIMIZATION
X = news[,-59]   # data (excluding the response variable)

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 10, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 9 ideal number of clusters
model1 <- Mclust(X, G = 9, modelNames = mclust.options("emModelNames"))

model1$classification

#### ### TASK 2 : DIMENSIONALITY REDUCTION TECHNIQUES

### Feature selection algorithm

news_Feature<-read.csv("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_4/data.csv")
news_Feature$shares <- as.factor(ifelse(news_Feature$shares < median(news_Feature$shares),0,1))
model_RF <- randomForest(x=news_Feature[,1:58], y= news_Feature$shares)
importance    <- importance(model_RF)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

#Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

#Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few() 
rank <- rankImportance[order(rankImportance$Rank),]
req_colns <- which(colnames(news_Feature) %in% rank$Variables[1:15])
Feat_sel_data <- cbind(subset(news_Feature, select = c(req_colns)),shares=news_Feature$shares)

##PCA
news_pca<-read.csv("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_4/data.csv")
news_pca<-news_pca[,-59]

pca_result <- prcomp(news_pca, scale = TRUE)
names(pca_result)

VE <- pca_result$sdev^2
PVE <- VE / sum(VE)
round(PVE,2)

PVEplot <- qplot(c(1:58), PVE) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("PVE") +
  ggtitle("Scree Plot") +
  ylim(0, 1)
cumPVE <- qplot(c(1:58), cumsum(PVE)) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab(NULL) + 
  ggtitle("Cumulative Scree Plot") +
  ylim(0,1)
grid.arrange(PVEplot, cumPVE, ncol = 2)
comp <- data.frame(pca_result$x[,1:25])

####ICA###
set.seed(123)
news_ICA <- news[,-59]
ICA_1 <- fastICA(news_ICA,58)
ICA_1_f <- data.frame(ICA_1$S)
ICA_1_final <- data.frame(a = names(ICA_1_f),k = kurtosis(ICA_1_f))

ICA_1_final <- ICA_1_final[(ICA_1_final$k < 5 & ICA_1_final$k > 2),]
num <- which(colnames(ICA_1_f) %in% ICA_1_final$a)
ICA_comp <- cbind(subset(ICA_1_f, select=c(num)),news$shares)
plot(ICA_1_final$k, ylab = "Kurtosis Score", main = "scatter plot of kurtosis values for \n selected Independent Components")

##Randomized projections
news_rpca<-read.csv("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_4/data.csv")
model_RPCA <- rpca(news_rpca[,1:58], k = NULL, center = TRUE, scale = TRUE) 
plot(model_RPCA$sdev^2, ylab = "Variance", type = "b", main = "plot of variance vs random projections")

RVE <- model_RPCA$sdev^2
RVE <- RVE / sum(RVE)
round(RVE,2)

RVEplot <- qplot(c(1:58), RVE) + 
  geom_line() + 
  xlab("Random Component") + 
  ylab("RVE") +
  ggtitle("Scree Plot") +
  ylim(0, 1)
cumRVE <- qplot(c(1:58), cumsum(RVE)) + 
  geom_line() + 
  xlab("Random Component") + 
  ylab(NULL) + 
  ggtitle("Cumulative Scree Plot") +
  ylim(0,1)
grid.arrange(RVEplot, cumRVE, ncol = 2)

RPCA_finaldata <- model_RPCA$x[,1:25]

#### ### TASK 3 : CLUSTERING AFTER DIMENSIONALITY REDUCTION

#### Kmeans of Feature selection data

Feat_sel_data_Kmeans <- Feat_sel_data[,-16]
##Function for elbow method
wss <- function(k) {
  kmeans(Feat_sel_data_Kmeans, k, nstart = 2 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 10
k.values <- 1:10

# extract wss for 2-10 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 10, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

### from the plot we can say that 4 would be the ideal number of clusters
## computing kmeans clustering with k = 4
final <- kmeans(Feat_sel_data_Kmeans, 4, nstart = 10)
print(final)

fviz_cluster(final, data = Feat_sel_data_Kmeans)

#### Kmeans of PCA data

##Function for elbow method
wss <- function(k) {
  kmeans(comp, k, nstart = 2 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 10
k.values <- 1:10

# extract wss for 2-10 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 10, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

### from the plot we can say that 9 would be the ideal number of clusters
## computing kmeans clustering with k = 9
final <- kmeans(comp, 9, nstart = 10)
print(final)

fviz_cluster(final, data = comp)

#### Kmeans of ICA data
ICA_comp_kmeans <- ICA_comp[,-12]
##Function for elbow method
wss <- function(k) {
  kmeans(ICA_comp_kmeans, k, nstart = 2 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

# extract wss for 2-10 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 10, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

### from the plot we can say that 10 would be the ideal number of clusters
## computing kmeans clustering with k = 10
final <- kmeans(ICA_comp_kmeans, 10, nstart = 10)
print(final)

fviz_cluster(final, data = ICA_comp_kmeans)

#### Kmeans of Random components data

##Function for elbow method
wss <- function(k) {
  kmeans(RPCA_finaldata, k, nstart = 2 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

# extract wss for 2-10 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 10, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

### from the plot we can say that 12 would be the ideal number of clusters
## computing kmeans clustering with k = 12
final <- kmeans(RPCA_finaldata, 12, nstart = 10)
print(final)

fviz_cluster(final, data = RPCA_finaldata)

### Expectation Maximization technique after Dimensionality Reduction
### EM on feature selection data
X = Feat_sel_data[,-16]   # data (excluding the response variable)

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 10, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 6 ideal number of clusters
model2 <- Mclust(X, G = 6, modelNames = mclust.options("emModelNames"))

model2$classification

### EM on PCA data
X = comp   # data (excluding the response variable)

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 15, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 8 ideal number of clusters
model3 <- Mclust(X, G = 8, modelNames = mclust.options("emModelNames"))
model3$classification

### EM on ICA data
X = ICA_comp[,-8]   # data (excluding the response variable)

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 15, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 10 ideal number of clusters
model4 <- Mclust(X, G = 10, modelNames = mclust.options("emModelNames"))
model4$classification

### EM on Random projections data
X = RPCA_finaldata   # data (excluding the response variable)

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 15, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 8 ideal number of clusters
model5 <- Mclust(X, G = 8, modelNames = mclust.options("emModelNames"))
model5$classification


######## TASK 4 NEURAL NETWORKS on DIMENSIONALITY REDUCTION DATA
###### NN of feature Selection Dataset

Feat_sel_data$shares <- as.numeric(Feat_sel_data$shares)

norm <- function(x){
  (x - min(x))/(max(x)-min(x))
}

Feat_sel_data <- cbind(data.frame(lapply(Feat_sel_data[,-16], norm)),shares = Feat_sel_data$shares)

NN_FS_news <- as.matrix(Feat_sel_data)
dimnames(NN_FS_news) <- NULL

set.seed(123)
ind <- sample(2, nrow(NN_FS_news), replace=TRUE, prob=c(0.67, 0.33))

NN_FS_news_train <- NN_FS_news[ind==1, 1:15]
NN_FS_news_test <- NN_FS_news[ind==2, 1:15]

# Split the class attribute
NN_FS_traintarget <- NN_FS_news[ind==1, 16]
NN_FS_testtarget <- NN_FS_news[ind==2, 16]

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(15)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model %>% fit(
  NN_FS_news_train,
  NN_FS_traintarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history)

# Predict the classes for the test data
classes <- model %>% predict_classes(NN_FS_news_test, batch_size = 128)

# Confusion matrix
table(NN_FS_testtarget, classes)

# Evaluate on test data and labels
score <- model %>% evaluate(NN_FS_news_test, NN_FS_testtarget, batch_size = 128)
print(score$acc)

###### NN of PCA Dataset
comp_final <- cbind(comp,news[,59])

comp_data <- cbind(data.frame(lapply(comp_final[,-26], norm)),shares = comp_final$`news[, 59]`)

comp_data <- as.matrix(comp_data)
dimnames(comp_data) <- NULL

set.seed(123)
ind <- sample(2, nrow(comp_data), replace=TRUE, prob=c(0.67, 0.33))

comp_train <- comp_data[ind==1, 1:25]
comp_test <- comp_data[ind==2, 1:25]

# Split the class attribute
comp_traintarget <- comp_data[ind==1, 26]
comp_testtarget <- comp_data[ind==2, 26]

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(25)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model %>% fit(
  comp_train,
  comp_traintarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history)

# Predict the classes for the test data
classes <- model %>% predict_classes(comp_test, batch_size = 128)

# Confusion matrix
table(comp_testtarget, classes)

# Evaluate on test data and labels
score <- model %>% evaluate(comp_test, comp_testtarget, batch_size = 128)
print(score$acc)

########## NN of ICA dataset

ICA_NN <- cbind(data.frame(lapply(ICA_comp[,-8], norm)),shares = ICA_comp$`news$shares`)

ICA_NN <- as.matrix(ICA_NN)
dimnames(ICA_NN) <- NULL

set.seed(123)
ind <- sample(2, nrow(ICA_NN), replace=TRUE, prob=c(0.67, 0.33))

ICA_train <- ICA_NN[ind==1, 1:7]
ICA_test <- ICA_NN[ind==2, 1:7]

# Split the class attribute
ICA_traintarget <- ICA_NN[ind==1, 8]
ICA_testtarget <- ICA_NN[ind==2, 8]

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(7)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model %>% fit(
  ICA_train,
  ICA_traintarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history)

# Predict the classes for the test data
classes <- model %>% predict_classes(ICA_test, batch_size = 128)

# Confusion matrix
table(ICA_testtarget, classes)

# Evaluate on test data and labels
score <- model %>% evaluate(ICA_test, ICA_testtarget, batch_size = 128)
print(score$acc)

########## NN of random projection dataset

RPCA_NN <- cbind(RPCA_finaldata,news[,59])

dimnames(RPCA_NN) <- NULL

set.seed(123)
ind <- sample(2, nrow(RPCA_NN), replace=TRUE, prob=c(0.67, 0.33))

RPCA_train <- RPCA_NN[ind==1, 1:25]
RPCA_test <- RPCA_NN[ind==2, 1:25]

# Split the class attribute
RPCA_traintarget <- RPCA_NN[ind==1, 26]
RPCA_testtarget <- RPCA_NN[ind==2, 26]

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(25)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model %>% fit(
  RPCA_train,
  RPCA_traintarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history)

# Predict the classes for the test data
classes <- model %>% predict_classes(RPCA_test, batch_size = 128)

# Confusion matrix
table(RPCA_testtarget, classes)

# Evaluate on test data and labels
score <- model %>% evaluate(RPCA_test, RPCA_testtarget, batch_size = 128)
print(score$acc)

################### TASK 5 ##############

task5 <- cbind(news,final$cluster,model1$classification)
task5_finaldata <- task5[,59:61]
names(task5_finaldata) <- c("y","kmeans", "EM")
task5_finaldata$kmeans <- as.character(task5_finaldata$kmeans)
task5_finaldata$EM <- as.character(task5_finaldata$EM)

##One hot encoding the variables
task5_y <- as.matrix(task5_finaldata[,1])

En_Kmeans<-stats::model.matrix(~kmeans, task5_finaldata)[,-1]
En_EM<-stats::model.matrix(~EM, task5_finaldata)[,-1]

Final_data <- cbind.data.frame(task5_y,En_Kmeans,En_EM)

## Neural Networks
NN_5 <- cbind(data.frame(lapply(Final_data[,-1], norm)),TargetBuy=Final_data$task5_y)

NN_5$TargetBuy <- as.numeric(NN_5$TargetBuy)


NN_5 <- as.matrix(NN_5)
dimnames(NN_5) <- NULL

set.seed(123)
ind <- sample(2, nrow(NN_5), replace=TRUE, prob=c(0.67, 0.33))

comp_train <- NN_5[ind==1, 1:15]
comp_test <- NN_5[ind==2, 1:15]

# Split the class attribute
comp_traintarget <- NN_5[ind==1, 16]
comp_testtarget <- NN_5[ind==2, 16]

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(15)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model %>% fit(
  comp_train,
  comp_traintarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history)

# Predict the classes for the test data
classes <- model %>% predict_classes(comp_test, batch_size = 128)

# Confusion matrix
table(comp_testtarget, classes)

# Evaluate on test data and labels
score <- model %>% evaluate(comp_test, comp_testtarget, batch_size = 128)
print(score$acc)


