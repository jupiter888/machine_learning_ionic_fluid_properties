from sklearn.ensemble import GradientBoostingRegressor
# Load data
from novy_load import data_loader
X_train_h, Y_train_h, X_test_h, Y_test_h = data_loader('DATA/hcap/cpjk1-train_data.xlsx','DATA/hcap/cpjk1-test_data.xlsx')

# Create Gradient Boosting Regressor model
gbriimodel_h = GradientBoostingRegressor()

# Convert col vector into 1-dim[] (ndarray)
from colvector_convert import ColVectorConverter
converter = ColVectorConverter(Y_train_h.values)
Y_train888 = converter.to_1d_array()
X_train888 = X_train_h.values

# Fit the model
gbriimodel_h.fit(X_train888, Y_train888)

# Predict on the train data
Y_pred_train_h = gbriimodel_h.predict(X_train888)
# Predict on the test data
Y_pred_test_h = gbriimodel_h.predict(X_test_h)

# Evaluate model on train and test
from eval_fx import eval_results
eval_results(model_h, "HCAP", "GRADIENT BOOSTING REGRESSOR", Y_train_888, Y_test_h, Y_pred_train_h, Y_pred_test_h)

