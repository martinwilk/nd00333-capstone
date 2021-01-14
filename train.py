from xgboost import XGBClassifier
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory



 


def clean_data(white_data, red_data):

    # Clean and one hot encode data
    x_white = white_data.to_pandas_dataframe().dropna()
    x_red = red_data.to_pandas_dataframe().dropna()
    x_white.loc[:,"wine_type"] = "WHITE"
    x_red.loc[:,"wine_type"] = "RED"
    x_df = pd.concat([x_white, x_red])
    x_df["is_red_wine"]=x_df.pop("wine_type").apply(lambda s: 1 if s=="RED" else 0)
    y_df = pd.cut(x_df.quality, bins=[1,4,6,9], labels=["BAD","MEDIUM","GOOD"])
    return x_df, y_df





def main():
    run = Run.get_context()
    ws = run.experiment.workspace
    found = False
    key = "wine-quality"
    description_text = "Wine Quality Dataset for Udacity Course 3"

    if key in ws.datasets.keys(): 
        found = True
        input_data = ws.datasets[key]
        features = input_data.to_pandas_dataframe()

    if not found:
        # Create AML Dataset and register it into Workspace
        url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
        url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        white_data = TabularDatasetFactory.from_delimited_files(url_white, separator=";")
        red_data = TabularDatasetFactory.from_delimited_files(url_red, separator=";")
        features, target = clean_data(white_data, red_data)
        features["quality"]=target
        ds = ws.get_default_datastore()
        input_data = TabularDatasetFactory.register_pandas_dataframe(dataframe=features, target=ds, name=key,
                                                                        description=description_text)


    target = features.pop("quality")
    target = target.replace({"BAD": -1, "MEDIUM": 0, "GOOD":1})
    x_train, x_test, y_train, y_test = train_test_split(features, target, random_state=0)
    # Add arguments to script
    parser = argparse.ArgumentParser()


    parser.add_argument("--max_depth", type=int, default=6, help="Maximum depth of tree")
    parser.add_argument("--alpha", type=float, default=0, help="L1 regularization")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")

    args = parser.parse_args()

    run.log("Max Depth:", np.int(args.max_depth))
    run.log("Alpha:", np.float(args.alpha))
    run.log("Learning rate:", np.float(args.learning_rate))


    model = XGBClassifier(booster="gbtree",
                          objective="multi:softmax",
                          subsample=0.6,
                          tree_method="auto",
                          n_estimators=250,
                          max_depth=args.max_depth,
                          reg_alpha=args.alpha,
                          learning_rate=args.learning_rate)
    model.fit(x_train, y_train)
    y_pred=model.predict_proba(x_test)
    auc = roc_auc_score(y_test, y_pred, average="weighted", multi_class="ovo", labels=model.classes_)

    os.makedirs("./outputs", exist_ok=True)
    joblib.dump(model, filename="./outputs/wine-quality-model.pkl")
    run.log("AUC_weighted", np.float(auc))

if __name__ == '__main__':
    main()
