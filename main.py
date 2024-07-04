from BaseLine import BaseLine
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from argparse import ArgumentParser




if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument('--data', type=str, required=True, help="path to the data set")
    # parser.add_argument('--target_feature', type=str, required=True, help="name of the target feature")
    # # parser.add_argument('--out', type=str, required=True, help="path of the output file as required in the task description")
    # args = parser.parse_args()
    path = "/Users/omercarbonne/Desktop/IML/projects/hackathon/preprocess_precent_5_of_datatrain_bus_schedule-2.csv"
    target_feature = "passengers_up"
    baseline = BaseLine()
    # baseline.fit(args.data, args.target_feature)
    baseline.fit(path, target_feature)

    baseline.show_predictions_graph()


