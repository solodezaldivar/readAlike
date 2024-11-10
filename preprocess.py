import pandas as pd
import numpy as np

from pandas import DataFrame


def read_data(path: str, cols: list[str]) -> DataFrame:
    return pd.read_csv(path, usecols=cols)


def drop_items_with_short_entries(df: DataFrame, col_names: list[str], limit: int = 10):
    for col_name in col_names:
        df = df[df[col_name].apply(lambda x: len(x.split()) >= limit)]


def drop_duplicates(df: DataFrame, col_name: str) -> DataFrame:
    return df.drop_duplicates(subset=col_name)


def convert_strings_into_lists(df: DataFrame, col_names: list[str]):
    for col_name in col_names:
        df[col_name] = df[col_name].apply(lambda x: x.strip("[]").replace("'", "").split(", "))


def preprocess_data(df: DataFrame) -> DataFrame:
    df.replace(r'', np.nan, regex=True, inplace=True)  # replace null with NaN
    df.dropna(inplace=True)  # drop items with NaN
    drop_items_with_short_entries(df, ['description'])
    df = drop_duplicates(df, 'Title')
    convert_strings_into_lists(df, ['authors', 'categories'])
    df.reset_index(drop=True, inplace=True)  # reset indices after dropping
    return df
