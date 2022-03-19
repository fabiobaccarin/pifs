"""
Creates a report with a profile of the dataset

Input: data/data_renamed.pkl
Output: reports/data_profile.html
"""

def main():
    import pandas as pd
    import pandas_profiling as pp
    from modules import paths

    out = paths.REPORTS.joinpath("data_profile.html")
    pp.ProfileReport(
        pd.read_pickle(paths.RENAMED_DATA),
        title="Profile of renamed dataset (raw)",
        minimal=True,
        progress_bar=False,
        correlations={"pearson": {"calculate": True, "threshold": 0.5}}
    ).to_file(out)

    assert out.is_file()

if __name__ == "__main__":
    main()