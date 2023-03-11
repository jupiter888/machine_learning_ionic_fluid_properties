from sklearn.tree import DecisionTreeRegressor
from novy_load import data_loader

# Load data
X_train_h, Y_train_h, X_test_h, Y_test_h = data_loader("DATA/hcap/cpjk1-train_data.xlsx", "DATA/hcap/cpjk1-test_data.xlsx")

# Create model
model_h = DecisionTreeRegressor()
# Fit model
model_h.fit(X_train_h, Y_train_h)

# Predict the test set labels(Make predictions on the X_test set)
Y_pred_test_h = model_h.predict(X_test_h)
# Predict the train set labels(Make predictions on the X_train set)
Y_pred_train_h = model_h.predict(X_train_h)

# Extensively evaluate model using function 
from eval_fx import eval_results
# RETURNS: data_set, model_name, train_mse, train_r2, train_mae, test_mse, test_r2, test_mae
eval_results(model_h, "HCAP", "DECISION TREE", Y_train_h, Y_test_h, Y_pred_train_h, Y_pred_test_h)

# Evaluate model hardcoding
#train_mse = mean_squared_error(Y_train_h, Y_pred_train_h)
#test_mse = mean_squared_error(Y_test_h, Y_pred_test_h)
#train_r2 = r2_score(Y_train_h, Y_pred_train_h)
#test_r2 = r2_score(Y_test_h,Y_pred_test_h)

#print predictions
#print(f"Train Predicted Data:\n{Y_pred_train_h}\n\n\n
#print(f"Test Predicted Data:\n{Y_pred_test_h}\n\n\n")

#print eval metrics
#print(f"Decision Tree\n Test MSE: {test_mse} Train MSE: {train_mse}\n Train R2: {train_r2} Test R2 {test_r2}\n")
