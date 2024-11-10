import pandas as pd
import numpy as np

from pandas import DataFrame

from config import DESCRIPTION_COL, TITLE_COL, AUTHORS_COL, CATEGORIES_COL


class Preprocessor:
    def __init__(self, path: str, cols: list[str]):
        self.df = pd.read_csv(path, usecols=cols)

    def drop_items_with_short_entries(self, col_names: list[str], limit: int = 10) -> None:
        for col_name in col_names:
            self.df = self.df[self.df[col_name].apply(lambda x: len(x.split()) >= limit)]

    def drop_duplicates(self, col_name: str) -> None:
        self.df = self.df.drop_duplicates(subset=col_name)

    def convert_strings_into_lists(self, col_names: list[str]):
        for col_name in col_names:
            self.df[col_name] = self.df[col_name].apply(lambda x: x.strip("[]").replace("'", "").split(", "))

    def preprocess_data(self) -> DataFrame:
        self.df.replace(r'', np.nan, regex=True, inplace=True)  # replace null with NaN
        self.df.dropna(inplace=True)  # drop items with NaN
        self.drop_items_with_short_entries([DESCRIPTION_COL])
        self.drop_duplicates(TITLE_COL)
        self.convert_strings_into_lists([AUTHORS_COL, CATEGORIES_COL])
        self.df.reset_index(drop=True, inplace=True)  # reset indices after dropping
        return self.df
