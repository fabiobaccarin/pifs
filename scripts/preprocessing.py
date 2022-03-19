"""
Applies preprocessing to data

Input: data/data_renamed.pkl
Outputs:
    - data/X.pkl
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
    X = df.filter(features)
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

    # Sanity checks
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[1] == len(features)
    assert all([x in features for x in X])
    assert len(X) == len(df)
    assert len(y) == len(df)
    assert y.name == "x00"
    assert all([X[x].min() == 0 and X[x].max() == 1 for x in X])
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert y.nunique() == 2

    # Saving data
    X.to_pickle(paths.DATA.joinpath("X.pkl"))
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