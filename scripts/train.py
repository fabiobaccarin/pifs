"""
Trains the model (LightGBM) on the training data X_train, y_train
"""

def main():
    import pandas as pd
    import joblib
    import lightgbm
    from modules import paths

    X_train = pd.read_pickle(paths.DATA.joinpath("X_train.pkl"))
    y_train = pd.read_pickle(paths.DATA.joinpath("y_train.pkl"))

    model = lightgbm.LGBMClassifier().fit(X_train, y_train)
    mp = paths.MODELS.joinpath("lgbm.pkl")
    joblib.dump(model, mp)

    # Sanity checks
    # Model can be loaded
    m = joblib.load(mp)

    # Model can predict
    m.predict_proba(X_train)

if __name__ == "__main__":
    main()