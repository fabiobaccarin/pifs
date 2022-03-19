"""
Does some cleaning in column names to make coding easier

Input: data/data.csv
Outputs:
    - data/colnames.json
    - data/data_renamed.pkl
"""

def main():
    import json
    import pandas as pd
    from modules import paths

    df = pd.read_csv(paths.RAW_DATA, encoding="utf-8")
    colnames = {c: f"x{i:02d}" for i, c in enumerate(df)}
    new = df.rename(colnames, axis="columns")

    # Checks that the shape remains the same
    assert new.shape == df.shape
    # Checks that columns were changed in the order intended
    cn = list(colnames.values())
    assert all([x in cn for x in new.columns.to_list()])

    json.dump(
        colnames,
        paths.DATA.joinpath("colnames.json").open("w"),
        indent=2,
    )
    new.to_pickle(paths.DATA.joinpath("data_renamed.pkl"))

if __name__ == "__main__":
    main()