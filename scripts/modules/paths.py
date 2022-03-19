"""
Contains paths to move around the project
"""

import pathlib

ROOT = pathlib.Path(__file__).parents[2]
DATA = ROOT.joinpath("data")
RAW_DATA = DATA.joinpath("data.csv")
RENAMED_DATA = DATA.joinpath("data_renamed.pkl")
REPORTS = ROOT.joinpath("reports")
FEATURES = DATA.joinpath("features.json")
MODELS = ROOT.joinpath("models")