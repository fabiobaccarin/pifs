"""
Splits data into train and test sets

Inputs:
    - data/X.pkl
    - data/X_selected.pkl
    - data/y.pkl
Outputs:
    - data/X_train.pkl
    - data/X_s_train.pkl
    - data/X_test.pkl
    - data/X_s_test.pkl
    - data/y_train.pkl
    - data/y_test.pkl
"""

# Proportions:
# train:    70%
# test:     30%

def main():
    import pandas as pd
    import math
    from modules import paths
    from sklearn import model_selection

    X = pd.read_pickle(paths.DATA.joinpath("X.pkl"))
    X_selected = pd.read_pickle(paths.DATA.joinpath("X_selected.pkl"))
    y = pd.read_pickle(paths.DATA.joinpath("y.pkl"))

    X_train, X_test, X_s_train, X_s_test, y_train, y_test = \
        model_selection.train_test_split(
            X, X_selected, y,
            test_size=0.3,
            random_state=0,
            stratify=y,
        )

    # Sanity checks
    REL_TOL = 0.01
    ABS_TOL = 0.02
    assert math.isclose(
        len(X_train) / len(X),
        0.7,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL
    )
    assert math.isclose(
        len(y_train) / len(y),
        0.7,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL
    )
    assert math.isclose(
        len(X_test) / len(X),
        0.30,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL
    )
    assert math.isclose(
        len(y_test) / len(y),
        0.30,
        rel_tol=REL_TOL,
        abs_tol=ABS_TOL
    )
    assert X_s_train.index.equals(X_train.index)
    assert X_s_test.index.equals(X_test.index)

    # Saving data
    X_train.to_pickle(paths.DATA.joinpath("X_train.pkl"))
    X_test.to_pickle(paths.DATA.joinpath("X_test.pkl"))
    X_s_train.to_pickle(paths.DATA.joinpath("X_s_train.pkl"))
    X_s_test.to_pickle(paths.DATA.joinpath("X_s_test.pkl"))
    y_train.to_pickle(paths.DATA.joinpath("y_train.pkl"))
    y_test.to_pickle(paths.DATA.joinpath("y_test.pkl"))


if __name__ == "__main__":
    main()