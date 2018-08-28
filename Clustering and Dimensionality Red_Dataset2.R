############# DATA SET 2: ORGANICS PRODUCT DATA ##################

###### NOTE: BOTH THE CODE AND REPORT ANALYSIS FOLLOW THE SAME ORDER, BRIEF COMMENTS ARE WRITTEN IN THE CODE TO UNDERSTAND WHATS BEING DONE ############

load.libraries <- c("tidyverse","cluster","factoextra", "ClusterR", "randomForest","ggthemes","rsvd","gridExtra", "mclust", "fastICA","moments","devtools","keras","tensorflow","class", "readxl")

install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs)
sapply(load.libraries, require, character = TRUE)

install_keras()
install_tensorflow()

### TASK 1 : Clustering Alogrithms

### K- MEANS

#importing the data and removed the non predictive variables
organics <- read_excel("C:/Users/Rahul/OneDrive/BA_Spring 2018 Sem/Applied Machine learning/HW_4/organics.xlsx")
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

org_kmeans <- organics_norm[,-31]

##Function for elbow method
wss <- function(k) {
  kmeans(org_kmeans, k, nstart = 2 )$tot.withinss
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
final <- kmeans(org_kmeans, 8, nstart = 10)
print(final)

fviz_cluster(final, data = org_kmeans)

##### EXPECTATION MAXIMIZATION
X = organics_norm[,-31]   # data (excluding the response variable)

y = organics_norm[,31]    # the response variable

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 10, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 8 ideal number of clusters
model1 <- Mclust(X, G = 8, modelNames = mclust.options("emModelNames"))

model1$classification

#### ### TASK 2 : DIMENSIONALITY REDUCTION TECHNIQUES

### Feature selection algorithm

org_Feature <- organics_norm
model_RF <- randomForest(x=org_Feature[,1:30], y= org_Feature$TargetBuy)
importance    <- importance(model_RF)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'IncNodePurity'],2))

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
req_colns <- which(colnames(org_Feature) %in% rank$Variables[1:4])
Feat_sel_data <- cbind(subset(org_Feature, select = c(req_colns)),TargetBuy=org_Feature$TargetBuy)

##PCA
org_pca <- organics_norm
org_pca<-org_pca[,-31]

pca_result <- prcomp(org_pca, scale = TRUE)
names(pca_result)

VE <- pca_result$sdev^2
PVE <- VE / sum(VE)
round(PVE,2)

PVEplot <- qplot(c(1:30), PVE) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("PVE") +
  ggtitle("Scree Plot") +
  ylim(0, 1)
cumPVE <- qplot(c(1:30), cumsum(PVE)) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab(NULL) + 
  ggtitle("Cumulative Scree Plot") +
  ylim(0,1)
grid.arrange(PVEplot, cumPVE, ncol = 2)
comp <- data.frame(pca_result$x[,1:17])

####ICA###
set.seed(123)
org_ICA <- organics_norm[,-31]
ICA_1 <- fastICA(org_ICA,29)
ICA_1_f <- data.frame(ICA_1$S)
ICA_1_final <- data.frame(a = names(ICA_1_f),k = kurtosis(ICA_1_f))

ICA_1_final <- ICA_1_final[(ICA_1_final$k < 5 & ICA_1_final$k > 1.5),]
num <- which(colnames(ICA_1_f) %in% ICA_1_final$a)
ICA_comp <- cbind(subset(ICA_1_f, select=c(num)),TargetBuy=organics_norm$TargetBuy)
plot(ICA_1_final$k, ylab = "Kurtosis Score", main = "scatter plot of kurtosis values for \n selected Independent Components")

##Randomized projections
org_rpca<-organics_norm
model_RPCA <- rpca(org_rpca[,1:30], k = NULL, center = TRUE, scale = TRUE) 
plot(model_RPCA$sdev^2, ylab = "Variance", type = "b", main = "plot of variance vs random projections")

RVE <- model_RPCA$sdev^2
RVE <- RVE / sum(RVE)
round(RVE,2)

RVEplot <- qplot(c(1:30), RVE) + 
  geom_line() + 
  xlab("Random Component") + 
  ylab("RVE") +
  ggtitle("Scree Plot") +
  ylim(0, 1)
cumRVE <- qplot(c(1:30), cumsum(RVE)) + 
  geom_line() + 
  xlab("Random Component") + 
  ylab(NULL) + 
  ggtitle("Cumulative Scree Plot") +
  ylim(0,1)
grid.arrange(RVEplot, cumRVE, ncol = 2)

RPCA_finaldata <- model_RPCA$x[,1:18]

#### ### TASK 3 : CLUSTERING AFTER DIMENSIONALITY REDUCTION

#### Kmeans of Feature selection data

Feat_sel_data_Kmeans <- Feat_sel_data[,-5]
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

### from the plot we can say that 6 would be the ideal number of clusters
## computing kmeans clustering with k = 6
final <- kmeans(Feat_sel_data_Kmeans, 6, nstart = 10)
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
ICA_comp_kmeans <- ICA_comp[,-8]
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

### from the plot we can say that 8 would be the ideal number of clusters
## computing kmeans clustering with k = 8
final <- kmeans(ICA_comp_kmeans, 8, nstart = 10)
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
X = Feat_sel_data[,-5]   # data (excluding the response variable)

y = Feat_sel_data[,5]    # the response variable

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 10, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 4 ideal number of clusters
model2 <- Mclust(X, G = 4, modelNames = mclust.options("emModelNames"))

model2$classification

### EM on PCA data
X = comp   # data (excluding the response variable)

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 15, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 10 ideal number of clusters
model3 <- Mclust(X, G = 10, modelNames = mclust.options("emModelNames"))
model3$classification

### EM on ICA data
X = ICA_comp[,-8]   # data (excluding the response variable)

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 15, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 7 ideal number of clusters
model4 <- Mclust(X, G = 7, modelNames = mclust.options("emModelNames"))
model4$classification

### EM on Random projections data
X = RPCA_finaldata   # data (excluding the response variable)

opt_gmm = Optimal_Clusters_GMM(X, max_clusters = 15, criterion = "BIC", 
                               
                               dist_mode = "eucl_dist", seed_mode = "random_subset",
                               
                               em_iter = 10, var_floor = 1e-10, 
                               
                               plot_data = T)

## 9 ideal number of clusters
model5 <- Mclust(X, G = 9, modelNames = mclust.options("emModelNames"))
model5$classification


######## TASK 4 NEURAL NETWORKS on DIMENSIONALITY REDUCTION DATA
###### NN of feature Selection Dataset

Feat_sel_data$TargetBuy <- as.numeric(Feat_sel_data$TargetBuy)

norm <- function(x){
  (x - min(x))/(max(x)-min(x))
}

Feat_sel_data <- cbind(data.frame(lapply(Feat_sel_data[,-5], norm)),TargetBuy= Feat_sel_data$TargetBuy)

NN_FS_org <- as.matrix(Feat_sel_data)
dimnames(NN_FS_org) <- NULL

set.seed(123)
ind <- sample(2, nrow(NN_FS_org), replace=TRUE, prob=c(0.67, 0.33))

NN_FS_org_train <- NN_FS_org[ind==1, 1:4]
NN_FS_org_test <- NN_FS_org[ind==2, 1:4]

# Split the class attribute
NN_FS_traintarget <- NN_FS_org[ind==1, 5]
NN_FS_testtarget <- NN_FS_org[ind==2, 5]

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(4)) %>% 
  layer_dense(units = 10, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'accuracy'
)

history <- model %>% fit(
  NN_FS_org_train,
  NN_FS_traintarget, 
  epochs = 20, 
  batch_size = 5, 
  validation_split = 0.2
)

plot(history)

# Predict the classes for the test data
classes <- model %>% predict_classes(NN_FS_org_test, batch_size = 128)

# Confusion matrix
table(NN_FS_testtarget, classes)

# Evaluate on test data and labels
score <- model %>% evaluate(NN_FS_org_test, NN_FS_testtarget, batch_size = 128)
print(score$acc)

###### NN of PCA Dataset
comp_final <- cbind(comp,organics_norm[,31])

comp_data <- cbind(data.frame(lapply(comp_final[,-18], norm)),TargetBuy=comp_final$`organics_norm[, 31]`)

comp_data <- as.matrix(comp_data)
dimnames(comp_data) <- NULL

set.seed(123)
ind <- sample(2, nrow(comp_data), replace=TRUE, prob=c(0.67, 0.33))

comp_train <- comp_data[ind==1, 1:17]
comp_test <- comp_data[ind==2, 1:17]

# Split the class attribute
comp_traintarget <- comp_data[ind==1, 18]
comp_testtarget <- comp_data[ind==2, 18]

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(17)) %>% 
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

ICA_NN <- cbind(data.frame(lapply(ICA_comp[,-8], norm)),TargetBuy= ICA_comp$TargetBuy)

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

RPCA_NN <- cbind(RPCA_finaldata,organics_norm[,31])

dimnames(RPCA_NN) <- NULL

set.seed(123)
ind <- sample(2, nrow(RPCA_NN), replace=TRUE, prob=c(0.67, 0.33))

RPCA_train <- RPCA_NN[ind==1, 1:18]
RPCA_test <- RPCA_NN[ind==2, 1:18]

# Split the class attribute
RPCA_traintarget <- RPCA_NN[ind==1, 19]
RPCA_testtarget <- RPCA_NN[ind==2, 19]

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(18)) %>% 
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


############### TASK 5 ############
task5 <- cbind(organics_norm,final$cluster,model1$classification)
task5_finaldata <- task5[,31:33]
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

comp_train <- NN_5[ind==1, 1:14]
comp_test <- NN_5[ind==2, 1:14]

# Split the class attribute
comp_traintarget <- NN_5[ind==1, 15]
comp_testtarget <- NN_5[ind==2, 15]

model <- keras_model_sequential()

model %>% 
  layer_dense(units = 20, activation = 'relu', input_shape = c(14)) %>% 
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