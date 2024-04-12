# Load required libraries
library(xgboost)
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

# Print message indicating dataset and target variables are defined successfully
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

# Train XGBoost for motor_UPDRS with LASSO selected features
cat("Training XGBoost for motor_UPDRS...\n")
xgb_model_motor <- xgboost(data = as.matrix(X_train_lasso_motor), 
                           label = Y_train_motor,
                           nrounds = 100, objective = "reg:squarederror")

# Print message indicating training XGBoost for motor_UPDRS is completed
cat("Training XGBoost for motor_UPDRS completed.\n")

# Make predictions for motor_UPDRS with XGBoost
cat("Making predictions for motor_UPDRS...\n")
predictions_motor_xgboost <- predict(xgb_model_motor, as.matrix(X_test_lasso_motor))

# Calculate evaluation metrics for motor_UPDRS
rmse_motor_xgboost <- sqrt(mean((predictions_motor_xgboost - Y_test_motor)^2))
r_squared_motor_xgboost <- cor(predictions_motor_xgboost, Y_test_motor)^2
mae_motor_xgboost <- mean(abs(predictions_motor_xgboost - Y_test_motor))

# Print results for motor_UPDRS
cat("\nMotor UPDRS :\n")
cat("RMSE:", rmse_motor_xgboost, "\n")
cat("R-squared:", r_squared_motor_xgboost, "\n")
cat("MAE:", mae_motor_xgboost, "\n\n")

# Save the trained model for motor_UPDRS
saveRDS(xgb_model_motor, "lasso_xgb_model_motor.rds")
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

# Train XGBoost for total_UPDRS with LASSO selected features
cat("Training XGBoost for total_UPDRS...\n")
xgb_model_total <- xgboost(data = as.matrix(X_train_lasso_total), 
                           label = Y_train_total,
                           nrounds = 100, objective = "reg:squarederror")

# Print message indicating training XGBoost for total_UPDRS is completed
cat("Training XGBoost for total_UPDRS completed.\n")

# Make predictions for total_UPDRS with XGBoost
cat("Making predictions for total_UPDRS...\n")
predictions_total_xgboost <- predict(xgb_model_total, as.matrix(X_test_lasso_total))

# Calculate evaluation metrics for total_UPDRS
rmse_total_xgboost <- sqrt(mean((predictions_total_xgboost - Y_test_total)^2))
r_squared_total_xgboost <- cor(predictions_total_xgboost, Y_test_total)^2
mae_total_xgboost <- mean(abs(predictions_total_xgboost - Y_test_total))

# Print results for total_UPDRS
cat("\nTotal UPDRS :\n")
cat("RMSE:", rmse_total_xgboost, "\n")
cat("R-squared:", r_squared_total_xgboost, "\n")
cat("MAE:", mae_total_xgboost, "\n\n")

# Save the trained model for total_UPDRS
saveRDS(xgb_model_total, "lasso_xgb_model_total.rds")
cat("\nModel for total_UPDRS saved.\n\n")
