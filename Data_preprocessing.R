library(dplyr)
library(randomForest)
library(caret)
library(e1071)
library(caTools)
library(corrplot)
library(Hmisc)
library(rpart)
library(xgboost)

# Load the Parkinson's dataset
parkinsons_data <- read.csv("parkinsons_updrs.csv")

# Define min-max scaling function
min_max_scaling <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Apply min-max scaling to the dataset
scaled_parkinsons_data <- parkinsons_data %>%
  mutate(across(-one_of("sex"), min_max_scaling))

# Shuffle and split the data into training and testing sets
set.seed(123)
split <- sample.split(scaled_parkinsons_data, SplitRatio = 0.7)
train_data <- scaled_parkinsons_data[split, ]
test_data <- scaled_parkinsons_data[!split, ]

# Write the scaled dataset to a CSV file
write.csv(scaled_parkinsons_data, "scaled_parkinsons_data.csv", row.names = FALSE)

# Compute the correlation matrix
correlation_matrix <- cor(scaled_parkinsons_data)

# Visualize the correlation matrix
corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust", 
         addrect = 2, tl.col = "black", tl.srt = 40, title = "Correlation Heatmap")

# Perform k-fold cross-validation (k = 5)
train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE, allowParallel = TRUE)

# Define the model training formula
formula <- as.formula("total_UPDRS ~ .")

# Decision Tree
tree_model <- rpart(formula, data = train_data, method = "anova")
tree_predictions <- predict(tree_model, newdata = test_data)
tree_rmse <- sqrt(mean((tree_predictions - test_data$total_UPDRS)^2))
tree_r_squared <- cor(tree_predictions, test_data$total_UPDRS)^2
tree_mae <- mean(abs(tree_predictions - test_data$total_UPDRS))
cat("\nThe metrics for Decision Tree are:\n")
cat("RMSE:", tree_rmse, "\n")
cat("R-Squared:", tree_r_squared, "\n")
cat("MAE:", tree_mae, "\n")

# Random Forest
rf_model_tuned <- train(
  formula, 
  data = train_data, 
  method = "rf", 
  tuneLength = 10, 
  trControl = train_control
)
best_rf_params <- rf_model_tuned$bestTune
rf_model_final <- randomForest(
  formula, 
  data = train_data, 
  ntree = 100,    # Reduced number of trees
  mtry = best_rf_params$mtry
)
rf_predictions <- predict(rf_model_final, newdata = test_data)
rf_rmse <- sqrt(mean((rf_predictions - test_data$total_UPDRS)^2))
rf_r_squared <- cor(rf_predictions, test_data$total_UPDRS)^2
rf_mae <- mean(abs(rf_predictions - test_data$total_UPDRS))
cat("\nThe metrics for Random Forest are:\n")
cat("RMSE:", rf_rmse, "\n")
cat("R-Squared:", rf_r_squared, "\n")
cat("MAE:", rf_mae, "\n")

# SVM
svm_model <- svm(formula, data = train_data, kernel="linear")
svm_predictions <- predict(svm_model, newdata = test_data)
svm_rmse <- sqrt(mean((svm_predictions - test_data$total_UPDRS)^2))
svm_r_squared <- cor(svm_predictions, test_data$total_UPDRS)^2
svm_mae <- mean(abs(svm_predictions - test_data$total_UPDRS))
cat("\nThe metrics for SVM are:\n")
cat("RMSE:", svm_rmse, "\n")
cat("R-Squared:", svm_r_squared, "\n")
cat("MAE:", svm_mae, "\n")

# XGBoost
xgb_model <- xgboost(data = as.matrix(train_data[-22]),
                     label = train_data$total_UPDRS,
                     nrounds = 500,
                     objective = "reg:squarederror",
                     verbose = FALSE)
xgb_predictions <- predict(xgb_model, newdata = as.matrix(test_data[-22]))
xgb_rmse <- sqrt(mean((xgb_predictions - test_data$total_UPDRS)^2))
xgb_r_squared <- cor(xgb_predictions, test_data$total_UPDRS)^2
xgb_mae <- mean(abs(xgb_predictions - test_data$total_UPDRS))
cat("\nThe metrics for XGBoost are:\n")
cat("RMSE:", xgb_rmse, "\n")
cat("R-Squared:", xgb_r_squared, "\n")
cat("MAE:", xgb_mae, "\n")

# Variable Importance Scores
importance_scores <- importance(rf_model_final)
sorted_importance <- importance_scores[order(-importance_scores[,1]), , drop = FALSE]
cat("\nVariable Importance Scores (in decreasing order):\n")
print(sorted_importance)

# Compute the correlation matrix
cor_data <- cor(train_data)

# Plot the correlation matrix
corrplot(cor_data, method = "number", number.cex = 0.7, addCoef.col = "black", tl.srt = 45)

# Identify highly correlated parameters
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
