"""
Reproduces the whole analysis for convenience. It also has logging behavior
"""

if __name__ == "__main__":
    import colnames
    import data_profile
    import corr_analysis
    import preprocessing
    import split

    colnames.main()
    data_profile.main()
    corr_analysis.main()
    preprocessing.main()
    split.main()