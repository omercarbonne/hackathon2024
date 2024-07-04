import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

SEED = 0
TEST_SIZE = 0.25


class LinearRegModel:

    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None
        self.target_feature = None

    def fit(self, path: str, target_feature: str) -> None:
        self.target_feature = target_feature
        df = pd.read_csv(path, encoding="ISO-8859-8")
        X, y = df.drop(target_feature, axis=1), df[target_feature]
        np.random.seed(SEED)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=TEST_SIZE)
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def loss(self):
        res = self.model.predict(self.X_test)
        res = res - self.y_test
        return np.mean(res ** 2)

    def show_predictions_graph(self):
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
        # todo
        # plt.fill_between(range(10, 101), loss_average - 2 * loss_std, loss_average + 2 * loss_std, alpha=0.2,
        #                  color="blue", label="error")
        plt.fill_between(list(range(10, 101)), loss_average - 2 * loss_std, loss_average + 2 * loss_std, alpha=0.2,
                         color="blue", label="error")

        plt.title("average loss")
        plt.xlabel("sample percentage")
        plt.ylabel('loss average')
        plt.grid(True)
        plt.savefig("loss.png")
        plt.clf()


    def get_sample(self,X: pd.DataFrame, y: pd.Series,  per: int) -> (pd.DataFrame, pd.Series):
        combine = pd.concat([X, y], axis=1)
        combine = combine.sample(frac=(per / 100))
        return combine.drop(self.target_feature, axis=1), combine[self.target_feature]

    def predict(self, x):
        return self.model.predict(x)












