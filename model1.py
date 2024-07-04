import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

SEED = 0


class LinearRegModel:

    def __init__(self, test_size=0.25):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.target_feature = None
        self.test_size = test_size
        self.name = None

    def fit(self, path: str, target_feature: str) -> None:
        self.target_feature = target_feature
        df = pd.read_csv(path, encoding="ISO-8859-8")
        X, y = df.drop(target_feature, axis=1), df[target_feature]
        np.random.seed(SEED)
        if self.test_size > 0:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = X, None, y, None
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.name = os.path.basename(path)

    def loss(self):
        res = self.model.predict(self.X_test)
        res = res - self.y_test
        return np.mean(res ** 2)


    def train_loss_graph(self):
        if self.test_size == 0:
            return
        loss_average = list()
        loss_std = list()
        for i in range(10, 101):
            loss_list = list()
            sum = 0
            for j in range(10):
                X_sample, y_sample = self.get_sample(self.X_train, self.y_train, i)
                self.model.fit(X_sample, y_sample)
                l = self.loss()
                loss_list.append(l)
                sum += l
            loss_average.append(sum / 10)
            loss_std.append(np.array(loss_list).reshape(-1, 1).std())

        percentage = pd.Series(range(10, 101))
        loss_average = pd.Series(loss_average)
        loss_std = pd.Series(loss_std)
        plt.scatter(percentage, loss_average, s=3)
        plt.fill_between(range(10, 101), loss_average - 2 * loss_std, loss_average + 2 * loss_std, alpha=0.2,
                         color="blue", label="error")
        plt.title("average loss")
        plt.xlabel("sample percentage")
        plt.ylabel('loss average')
        plt.grid(True)
        plt.savefig(self.name + "_train_loss.png")
        plt.clf()

    def predict(self, x):
        predictions = self.model.predict(x)
        return np.clip(predictions, a_min=0, a_max=None)
    def compare_predictions_graph(self):
        if self.test_size == 0:
            return
        pred_y = self.predict(self.X_test)
        plt.figure(figsize=(8, 6))
        plt.scatter(pred_y, self.y_test, color='blue', marker='o', label='Actual vs Predicted')

        # Adding a diagonal line for perfect prediction comparison
        plt.plot(pred_y, pred_y, color='red', linestyle='--', label='Perfect Prediction')

        # Adding labels and title
        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Scatter Plot of Predicted vs Actual Values')

        # Adding legend and grid
        plt.legend()
        plt.grid(True)

        # Display the plot
        plt.savefig(self.name + "_predictions_accuracy.png")


    def get_sample(self,X: pd.DataFrame, y: pd.Series,  per: int) -> (pd.DataFrame, pd.Series):
        combine = pd.concat([X, y], axis=1)
        combine = combine.sample(frac=(per / 100))
        return combine.drop(self.target_feature, axis=1), combine[self.target_feature]

















