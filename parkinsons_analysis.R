# parkinsons data set ----

library(dplyr)

parkinsons_data <- read.csv("parkinsons_updrs.csv")

min_max_scaling <- function(x) {
  (x - min(x)) / (max(x) - min(x))}

columns_to_exclude <- c("sex")  

scaled_parkinsons_data <- parkinsons_data %>%
  mutate(across(-one_of(columns_to_exclude), min_max_scaling))

tail(scaled_parkinsons_data)

random_values <- scaled_parkinsons_data %>%
  sample_n(5)

print(random_values)

write.csv(scaled_parkinsons_data, "scaled_parkinsons_data.csv", row.names = FALSE)


library(corrplot)

#scaled parkinson data set ----

correlation_matrix <- cor(scaled_parkinsons_data)

corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust", 
         addrect = 2, tl.col = "black", tl.srt = 40, title = "Correlation Heatmap")

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
cat("\n The metrics for Decision Tree regressor are : \n")
cat("RMSE:", rmse)

ssr <- sum((predictions - mean(test_data$total_UPDRS))^2)
sst <- sum((test_data$total_UPDRS - mean(test_data$total_UPDRS)^2))
R2 <- 1 - ssr/sst
cat("\nR-squared (R2):", R2)
mae <- mean(abs(predictions - test_data$total_UPDRS))
cat("\nMean Absolute Error (MAE):", mae)


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

# XGBoost 
library(xgboost)
set.seed(123)
xgb_model <- xgboost(data = as.matrix(train[-22]),
                     label = train$total_UPDRS,
                     nrounds = 500,
                     objective = "reg:squarederror",
                     verbose = FALSE)

xgb_pred <- predict(xgb_model, newdata = as.matrix(test[-22]))

xgb_rmse <- sqrt(mean((xgb_pred - test$total_UPDRS)^2))
xgb_r_squared <- cor(xgb_pred, test$total_UPDRS)^2
xgb_mae <- mean(abs(xgb_pred - test$total_UPDRS))

cat("\n\nThe metrics for XGBoost are:\n")
cat("RMSE:", xgb_rmse)
cat("\nR-Squared:", xgb_r_squared)
cat("\nMAE:", xgb_mae)


# Imp scores
importance_scores <- importance(classifier_RF)

sorted_importance <- importance_scores[order(-importance_scores[,1]), ,
                                       drop = FALSE]

cat("\n\n\nVariable Importance Scores (in decreasing order):\n")
print(sorted_importance)

cor_data <- cor(data)

#print("Correlation Matrix")
#print(cor_data)

library(Hmisc)

p_values <- rcorr(as.matrix(data))

library(corrplot)

corrplot(cor_data, method = "number", number.cex = 0.7, addCoef.col = "black", tl.srt = 45)

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

