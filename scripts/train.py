"""
Trains the models (LightGBM) on the training data

Inputs:
    - data/X_train.pkl
    - data/X_s_train.pkl
    - data/y_train.pkl
Outputs:
    - models/lgbm.pkl
    - models/lgbm_selected.pkl
"""

def main():
    import pandas as pd
    import joblib
    import lightgbm
    from modules import paths

    X_train = pd.read_pickle(paths.DATA.joinpath("X_train.pkl"))
    X_s_train = pd.read_pickle(paths.DATA.joinpath("X_s_train.pkl"))
    y_train = pd.read_pickle(paths.DATA.joinpath("y_train.pkl"))

    model = lightgbm.LGBMClassifier().fit(X_train, y_train)
    model_selected = lightgbm.LGBMClassifier().fit(X_s_train, y_train)
    mp = paths.MODELS.joinpath("lgbm.pkl")
    mp_selected = paths.MODELS.joinpath("lgbm_selected.pkl")
    joblib.dump(model, mp)
    joblib.dump(model_selected, mp_selected)

    # Sanity checks
    # Model can be loaded
    m = joblib.load(mp)
    ms = joblib.load(mp_selected)

    # Model can predict
    m.predict(X_train)
    ms.predict(X_s_train)

if __name__ == "__main__":
    main()