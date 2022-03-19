"""
Generates reports from correlation analysis

Input: data/data_renamed.pkl
Outputs:
    - reports/01_correlations_target.html
    - data/top5.json
    - data/features.json
"""

TOP5 = ["x86", "x37", "x90", "x84", "x17"]
TARGET = "x00"
CUTOFF = 0.5

def rep01(df, fp):
    """Creates report 01: correlations with target"""
    (df.drop("x00", axis=1)
        .corrwith(df["x00"])
        .sort_values(ascending=False, key=lambda x: x.abs())
        .to_frame()
        .rename({0: "Correlation with target"}, axis="columns")
        .to_html(fp, float_format="{:.2%}".format))

def feature_list(df, cutoff, target): # -> List[str]
    """Creates the feature list. See correlation.ipynb for details"""
    corr = df.drop(target, axis=1).corr().gt(cutoff)
    features = [v for v in TOP5]
    for x in df.drop(TOP5 + [target], axis=1):
        if corr.loc[x, features].any():
            continue
        else:
            features.append(x)
    return features

def main():
    import pandas as pd
    import json
    from modules import paths

    # x85 has +99% nulls and x94 is constant
    df = pd.read_pickle(paths.RENAMED_DATA).drop(["x85", "x94"], axis=1)

    rep01(df, paths.REPORTS.joinpath("01_correlations_target.html"))

    # Save top5
    json.dump(TOP5, paths.DATA.joinpath("top5.json").open("w"), indent=2)

    # Features
    fl = feature_list(df, CUTOFF, TARGET)

    # Sanity checks
    assert len(fl) >= 5
    assert all([x in fl for x in TOP5])

    # Save feature list
    json.dump(fl, paths.DATA.joinpath("features.json").open("w"), indent=2)

if __name__ == "__main__":
    main()