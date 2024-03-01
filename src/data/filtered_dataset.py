from enum import Enum
from typing import Literal, Union

import pandas as pd
import dask.dataframe as dd

class Level(Enum):
    CATEGORY = 'category'
    ATTACK = 'attack'

class Mode(Enum):
    EXC = 'exc'
    INC = 'inc'

class FilteredDataset:
    op = {Mode.EXC: lambda a, b: a != b, Mode.INC: lambda a, b: a == b}
    DataFrame = Union[pd.DataFrame, dd.DataFrame]

    def __init__(self, dataset: DataFrame, category_col_name: str, attack_col_name: str) -> None:
        self._dataset = dataset
        self.col_labels = {Level.CATEGORY: category_col_name, Level.ATTACK: attack_col_name}


    def get_variant(self, level: Level, mode: Mode, label:str) -> DataFrame:
        return self._dataset[self.op[mode](self._dataset[self.col_labels[level]], label)]

    def get_variants(self, level: Level, mode: Mode) -> DataFrame:
        return [self.get_variant(level, mode, label) for label in self._dataset[self.col_labels[level]].unique()]

    def get_all_variants(self) -> DataFrame:
        return zip(*[(self.get_variants(l, m), (l,m)) for m in Mode for l in Level])
