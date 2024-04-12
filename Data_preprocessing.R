#### Decision Tree ####
library(dplyr)
library(e1071)
library(rpart)

data <- read.csv("scaled_parkinsons_data.csv")

set.seed(123)
train_index <- sample(1:nrow(data), 0.7*nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

tree_model <- rpart(total_UPDRS ~ ., data=train_data, method = "anova")

predictions <- predict(tree_model, test_data)

rmse <- sqrt(mean((predictions - test_data$total_UPDRS)^2))
cat("\n The metrics for decision tree regressor are : \n\n")
print(paste("RMSE:", rmse))

ssr <- sum((predictions - mean(test_data$total_UPDRS))^2)
sst <- sum((test_data$total_UPDRS - mean(test_data$total_UPDRS)^2))
R2 <- 1 - ssr/sst
print(paste("R-squared (R2):", R2))

mae <- mean(abs(predictions - test_data$total_UPDRS))
print(paste("Mean Absolute Error (MAE):", mae))


#### Random Forest ####
library(randomForest)
library(caTools)

split <- sample.split(data, SplitRatio = 0.8)
split

train <- subset(data, split == "TRUE")
test <- subset(data, split == "FALSE")

set.seed(120)
classifier_RF <- randomForest(x=train[-22],
                              y=train$total_UPDRS,
                              ntree = 500)

classifier_RF

y_pred <- predict(classifier_RF, newdata = test[-22])

confusion_mtx <- table(test[, 22], y_pred)
confusion_mtx

plot(classifier_RF)

importance(classifier_RF)

varImpPlot(classifier_RF)

rf_rmse <- sqrt(mean((y_pred - test$total_UPDRS)^2))

rf_r_squared <- cor(y_pred, test$total_UPDRS)^2

rf_mae <- mean(abs(y_pred - test$total_UPDRS))

cat("\n\nThe metrics for Random Forest are:\n")
cat("RMSE:",rf_rmse)
cat("\nR-Squared:",rf_r_squared)
cat("\nMAE:",rf_mae)


#### SVM ####
svm_model <- svm(total_UPDRS ~ ., data = train, kernel="linear")

svm_pred <- predict(svm_model, newdata = test)

svm_rmse <- sqrt(mean((svm_pred - test$total_UPDRS)^2))

svm_r_squared <- cor(svm_pred, test$total_UPDRS)^2

svm_mae <- mean(abs(svm_pred - test$total_UPDRS))

cat("\n\nThe metrics for SVM are:")
cat("\nRMSE:", svm_rmse)
cat("\nR-squared:", svm_r_squared)
cat("\nMean Absolute Error:", svm_mae)

importance_scores <- importance(classifier_RF)

sorted_importance <- importance_scores[order(-importance_scores[,1]), ,
                                       drop = FALSE]

cat("\nVariable Importance Scores (Decreasing Order):\n")
print(sorted_importance)

cor_data <- cor(data)

#print("Correlation Matrix")
#print(cor_data)

library(Hmisc)

p_values <- rcorr(as.matrix(data))

library(corrplot)

corrplot(cor_data, method = "number")

cor_threshold <- which(abs(cor_data) > 0.5 & upper.tri(cor_data), arr.ind = TRUE)

cor_threshold <- cor_threshold[order(-abs(cor_data[cor_threshold])), , drop = FALSE]

param_names <- colnames(cor_data)

cat("\nHighly Correlated Parameters:\n")

for(i in 1:nrow(cor_data)){
  row_index <- cor_threshold[i,1]
  col_index <- cor_threshold[i,2]
  param1 <- param_names[row_index]
  param2 <- param_names[col_index]
  correlation <- cor_data[row_index, col_index]
  cat(param1, "and", param2, "have a correlation of", correlation, "\n")
}