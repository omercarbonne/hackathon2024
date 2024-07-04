
from argparse import ArgumentParser
import logging
from preprocess1 import *
from model1 import *


"""
usage:
    python code/main.py --training_set PATH --test_set PATH --out PATH

for example:
    python code/main.py --training_set /cs/usr/gililior/training.csv --test_set /cs/usr/gililior/test.csv --out predictions/trip_duration_predictions.csv 

"""


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    # 2. preprocess the training set
    logging.info("preprocessing train...")
    # preprocess_train = preprocess_train_task_1(args.training_set)
    preprocess_train = preprocess_train_task_1(args.training_set, False)
    target_feature = "passengers_up"

    # 3. train a model
    model = LinearRegModel()
    logging.info("training...")
    model.fit(preprocess_train, target_feature)

    # 4. load the test set (args.test_set)
    # 5. preprocess the test set
    logging.info("preprocessing test...")
    # preprocess_test = preprocess_train_task_1(args.test_set)
    preprocess_test, x_label = preprocess_text_task_1(args.test_set)
    test_set = pd.read_csv(preprocess_test, encoding="ISO-8859-8")
    x_label = pd.read_csv(x_label, encoding="ISO-8859-8")

    X_test = test_set.drop(target_feature, axis=1)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(X_test)

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    output = pd.DataFrame({
        'trip_id_unique_station': x_label['trip_id_unique_station'],
        'passengers_up': predictions.astype(int)})
    output.to_csv(args.out, index=False)