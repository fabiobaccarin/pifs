"""
Applies preprocessing to data

Input: data/data_renamed.pkl
Outputs:
    - data/X.pkl
    - data/X_selected.pkl
    - data/y.pkl
"""

import pandas as pd

def wrap(df, func, **kws): # -> DataFrame
    """Wraps the result of func(df) in a Dataframe"""
    return pd.DataFrame(
        func(df, **kws),
        index=df.index,
        columns=df.columns
    )

def main():
    import json
    import pandas_profiling as pp
    from sklearn import preprocessing
    from modules import paths

    df = pd.read_pickle(paths.RENAMED_DATA)
    features = json.load(paths.FEATURES.open("r"))
    X = df.drop(["x00", "x85", "x94"], axis=1)
    y = df["x00"].copy()

    # Simple transformations -- just enough to address adequately the point of
    # the project
    X = (
        X.pipe(
            wrap,
            preprocessing.quantile_transform,
            output_distribution="normal"
        )
        .pipe(wrap, preprocessing.minmax_scale)
    )

    # Feature selection: see reports/correlation.html
    X_selected = X.filter(features)

    # Sanity checks
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X.columns) == len(df.columns) - 3
    assert X_selected.shape[1] == len(features)
    assert all([x in features for x in X_selected])
    assert len(X) == len(df)
    assert len(y) == len(df)
    assert y.name == "x00"
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert y.nunique() == 2

    # Saving data
    X.to_pickle(paths.DATA.joinpath("X.pkl"))
    X_selected.to_pickle(paths.DATA.joinpath("X_selected.pkl"))
    y.to_pickle(paths.DATA.joinpath("y.pkl"))

    # Profile report on X
    pp.ProfileReport(
        X,
        title="Profile on preprocessed feature matrix",
        minimal=True,
        progress_bar=False,
    ).to_file(paths.REPORTS.joinpath("X_profile.html"))

if __name__ == "__main__":
    main()