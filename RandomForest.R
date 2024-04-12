# Load required libraries
library(randomForest)
library(caret)
library(glmnet)

# Print message indicating that libraries are loaded
cat("Libraries loaded successfully.\n")

# Read the data
scaled_parkinsons_data <- read.csv("scaled_parkinsons_data.csv")

# Print message indicating that data is loaded
cat("Data loaded successfully.\n")

# Define the dataset and target variables
X <- scaled_parkinsons_data[, -c(1, 4, 5, 6)]  # Exclude columns 1, 4, 5, and 6 as predictors
Y_motor <- scaled_parkinsons_data[, 5]   # Column 5 is the motor_UPDRS target variable
Y_total <- scaled_parkinsons_data[, 6]   # Column 6 is the total_UPDRS target variable

# Print message indicating dataset and target variables are defined
cat("Dataset and target variables defined successfully.\n")

# Perform train-test split for motor_UPDRS
set.seed(123) # for reproducibility
train_indices_motor <- createDataPartition(Y_motor, p = 0.8, list = FALSE)
X_train_motor <- X[train_indices_motor, ]
Y_train_motor <- Y_motor[train_indices_motor]
X_test_motor <- X[-train_indices_motor, ]
Y_test_motor <- Y_motor[-train_indices_motor]

# Print message indicating train-test split is completed for motor_UPDRS
cat("Train-test split completed for motor_UPDRS.\n")

# Perform feature selection using LASSO for motor_UPDRS
cat("Performing feature selection using LASSO for motor_UPDRS...\n")
lasso_model_motor <- cv.glmnet(as.matrix(X_train_motor), Y_train_motor, alpha = 1)
lasso_coef_motor <- coef(lasso_model_motor, s = "lambda.min")[-1,]
lasso_selected_indices_motor <- order(abs(lasso_coef_motor), decreasing = TRUE)[1:12] # Select top features
X_train_lasso_motor <- X_train_motor[, lasso_selected_indices_motor]
X_test_lasso_motor <- X_test_motor[, lasso_selected_indices_motor]

# Print selected features after LASSO for motor_UPDRS
cat("\nSelected features after LASSO for motor_UPDRS:\n")
selected_features_lasso_motor <- colnames(X_train_lasso_motor)
print(selected_features_lasso_motor)

# Append motor_UPDRS to the selected features data frame for motor_UPDRS
X_train_lasso_motor$motor_UPDRS <- Y_train_motor
#X_test_lasso_motor$motor_UPDRS <- Y_test_motor

# Print message indicating feature selection with LASSO for motor_UPDRS is completed
cat("Feature selection with LASSO for motor_UPDRS completed.\n")

# Define the hyperparameter grid for Random Forest for motor_UPDRS
grid_mtry_motor <- seq(1, ncol(X_train_lasso_motor) - 1)  # Exclude the target variable
#grid_ntree <- c(100, 200, 300,400,500,600) 
grid_motor <- expand.grid(mtry = grid_mtry_motor)

# Print message indicating hyperparameter grid is defined for motor_UPDRS
cat("Hyperparameter grid defined successfully for motor_UPDRS.\n")

# Define the training control for hyperparameter tuning
ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

# Print message indicating training control is defined
cat("Training control defined successfully.\n")

# Train Random Forest for motor_UPDRS with LASSO selected features
cat("Training Random Forest for motor_UPDRS...\n")
tryCatch({
  rf_motor_lasso <- train(motor_UPDRS ~ ., data = X_train_lasso_motor, method = "rf", trControl = ctrl, tuneGrid = grid_motor)
  best_mtry_motor_lasso <- rf_motor_lasso$bestTune$mtry
  
  # Print message indicating training Random Forest for motor_UPDRS with LASSO is completed
  cat("Training Random Forest for motor_UPDRS completed.\n")
}, error = function(e) {
  print("Error occurred during training Random Forest for motor_UPDRS:")
  print(e)
})

# Make predictions for motor_UPDRS with LASSO
cat("Making predictions for motor_UPDRS...\n")
predictions_motor_lasso <- predict(rf_motor_lasso, newdata = X_test_lasso_motor)
rmse_motor_lasso <- sqrt(mean((predictions_motor_lasso - Y_test_motor)^2))
r_squared_motor_lasso <- cor(predictions_motor_lasso, Y_test_motor)^2
mae_motor_lasso <- mean(abs(predictions_motor_lasso - Y_test_motor))

# Print results for motor_UPDRS
cat("\nMotor UPDRS :\n")
cat("RMSE:", rmse_motor_lasso, "\n")
cat("R-squared:", r_squared_motor_lasso, "\n")
cat("MAE:", mae_motor_lasso, "\n\n")

# Save the trained model for motor_UPDRS
saveRDS(rf_motor_lasso, "lasso_rf_model_motor.rds")
cat("\nModel for motor_UPDRS saved.\n\n")

# Perform train-test split for total_UPDRS
train_indices_total <- createDataPartition(Y_total, p = 0.8, list = FALSE)
X_train_total <- X[train_indices_total, ]
Y_train_total <- Y_total[train_indices_total]
X_test_total <- X[-train_indices_total, ]
Y_test_total <- Y_total[-train_indices_total]

# Print message indicating train-test split is completed for total_UPDRS
cat("Train-test split completed for total_UPDRS.\n")

# Perform feature selection using LASSO for total_UPDRS
cat("Performing feature selection using LASSO for total_UPDRS...\n")
lasso_model_total <- cv.glmnet(as.matrix(X_train_total), Y_train_total, alpha = 1)
lasso_coef_total <- coef(lasso_model_total, s = "lambda.min")[-1,]
lasso_selected_indices_total <- order(abs(lasso_coef_total), decreasing = TRUE)[1:12] # Select top features
X_train_lasso_total <- X_train_total[, lasso_selected_indices_total]
X_test_lasso_total <- X_test_total[, lasso_selected_indices_total]

# Print selected features after LASSO for total_UPDRS
cat("\nSelected features after LASSO for total_UPDRS:\n")
selected_features_lasso_total <- colnames(X_train_lasso_total)
print(selected_features_lasso_total)

# Append total_UPDRS to the selected features data frame for total_UPDRS
X_train_lasso_total$total_UPDRS <- Y_train_total
#X_test_lasso_total$total_UPDRS <- Y_test_total

# Print message indicating feature selection with LASSO for total_UPDRS is completed
cat("Feature selection with LASSO for total_UPDRS completed.\n")

# Define the hyperparameter grid for Random Forest for total_UPDRS
grid_mtry_total <- seq(1, ncol(X_train_lasso_total) - 1)  # Exclude the target variable
grid_total <- expand.grid(mtry = grid_mtry_total)

# Print message indicating hyperparameter grid is defined for total_UPDRS
cat("Hyperparameter grid defined successfully for total_UPDRS.\n")

# Train Random Forest for total_UPDRS with LASSO selected features
cat("Training Random Forest for total_UPDRS...\n")
tryCatch({
  rf_total_lasso <- train(total_UPDRS ~ ., data = X_train_lasso_total, method = "rf", trControl = ctrl, tuneGrid = grid_total)
  best_mtry_total_lasso <- rf_total_lasso$bestTune$mtry
  
  # Print message indicating training Random Forest for total_UPDRS with LASSO is completed
  cat("Training Random Forest for total_UPDRS completed.\n")
}, error = function(e) {
  print("Error occurred during training Random Forest for total_UPDRS:")
  print(e)
})

# Make predictions for total_UPDRS with LASSO
cat("Making predictions for total_UPDRS...\n")
predictions_total_lasso <- predict(rf_total_lasso, newdata = X_test_lasso_total)
rmse_total_lasso <- sqrt(mean((predictions_total_lasso - Y_test_total)^2))
r_squared_total_lasso <- cor(predictions_total_lasso, Y_test_total)^2
mae_total_lasso <- mean(abs(predictions_total_lasso - Y_test_total))

# Print results for total_UPDRS
cat("\nTotal UPDRS :\n")
cat("RMSE:", rmse_total_lasso, "\n")
cat("R-squared:", r_squared_total_lasso, "\n")
cat("MAE:", mae_total_lasso, "\n\n")

# Save the trained model for total_UPDRS
saveRDS(rf_total_lasso, "lasso_rf_model_total.rds")
cat("\nModel for total_UPDRS saved.\n\n")