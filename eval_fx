from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np


class EvalModelEvaluator:
    def __init__(self, model, dataset_name, model_name, Y_train, Y_test, Y_pred_train, Y_pred_test, ddd): #, df_9, df_preds6
        self.model = model
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.Y_pred_train = Y_pred_train
        self.Y_pred_test = Y_pred_test
        self.ddd = ddd

    def evaluate(self):
        # Evaluate the model's performance on the training data
        train_mse = mean_squared_error(self.Y_train, self.Y_pred_train)
        train_r2 = r2_score(self.Y_train, self.Y_pred_train)
        train_mae = mean_absolute_error(self.Y_train, self.Y_pred_train)
        # Evaluate models performance on test data
        test_mse = mean_squared_error(self.Y_test, self.Y_pred_test)
        test_r2 = r2_score(self.Y_test, self.Y_pred_test)
        test_mae = mean_absolute_error(self.Y_test, self.Y_pred_test)
        # Printing
        print(f"Train Predictions({self.dataset_name}):\n{self.Y_pred_train}\n-----------------------------------------------\nTest Predictions({self.dataset_name}):\n{self.Y_pred_test}\n\n\n-------------------------------------------------------------------------\n")
        print(f"PREDICTIONS OF ML MODEL {self.model_name} ON DATASET({self.dataset_name})")
        print(f"Training Predictions({self.dataset_name}):\n{self.Y_pred_train}\n-----------------------------------------------\nTesting Predictions({self.dataset_name}):\n{self.Y_pred_test}\n\n\n---------------------------------\n")
        print(f"{self.model_name} MODEL EVALUATION FOR {self.dataset_name} DATASET:\nTrain MSE: {train_mse} Train R2: {train_r2} Train MAE: {train_mae}\nTest MSE: {test_mse} Test R2: {test_r2} Test MAE: {test_mae}\n")
        print(f"[EVALUATION OF {self.model_name} WITH {self.dataset_name} DATA: COMPLETE]\n")

    def app_row(self):
        # Evaluate the model's performance on the training data
        train_mse = mean_squared_error(self.Y_train, self.Y_pred_train)
        train_r2 = r2_score(self.Y_train, self.Y_pred_train)
        train_mae = mean_absolute_error(self.Y_train, self.Y_pred_train)
        # Evaluate models performance on test data
        test_mse = mean_squared_error(self.Y_test, self.Y_pred_test)
        test_r2 = r2_score(self.Y_test, self.Y_pred_test)
        test_mae = mean_absolute_error(self.Y_test, self.Y_pred_test)

        bbb = pd.DataFrame(
            {"dataset_name": [self.dataset_name], "model_name": self.model_name,
             "train_r2": [train_r2], "train_mse": [train_mse], "train_mae": [train_mae],
             "test_r2": [test_r2], "test_mse": [test_mse], "test_mae": [test_mae]})

        dataset = [self.dataset_name] * (len(self.Y_pred_train))
        model_name = [self.model_name] * (len(self.Y_pred_train))
        na = len(self.Y_pred_train) - len(self.Y_pred_test)
        list_na = [np.nan] * na

        y_pred_test_list = np.append(self.Y_pred_test, list_na)

        new_row_df_preds = pd.DataFrame({"DataSet_Name": dataset,
                                         "Model Name": model_name,
                                         "Predict_Train": self.Y_pred_train, "Predict_Test": y_pred_test_list})

        ddd = pd.concat([self.ddd, bbb], axis=0, ignore_index=True)
        return new_row_df_preds, ddd

