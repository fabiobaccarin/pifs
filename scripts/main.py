"""
Reproduces the whole analysis for convenience. It also has logging behavior
"""

if __name__ == "__main__":
    import colnames
    import data_profile
    import preprocessing
    import split
    import train

    colnames.main()
    data_profile.main()
    preprocessing.main()
    split.main()
    train.main()