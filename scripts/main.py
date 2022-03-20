"""
Reproduces the whole analysis for convenience
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