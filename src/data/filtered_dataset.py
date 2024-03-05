from enum import Enum
from typing import Callable, Union

import pandas as pd
import dask.dataframe as dd

class Level(Enum):
    CATEGORY = 'category'
    ATTACK = 'attack'

class Mode(Enum):
    EXC = 'exc'
    INC = 'inc'

DataFrame = Union[pd.DataFrame, dd.DataFrame]

class FilteredDataset:
    op: dict[Mode, Callable[[str, str], bool]] = {Mode.EXC: lambda a, b: a != b, Mode.INC: lambda a, b: a == b}
    
    def __init__(self, dataset: DataFrame, category_col_name: str, attack_col_name: str) -> None:
        self._dataset: DataFrame = dataset
        self.col_labels = {Level.CATEGORY: category_col_name, Level.ATTACK: attack_col_name}


    def get_variant(self, level: Level, mode: Mode, label:str) -> DataFrame:
        return self._dataset[self.op[mode](self._dataset[self.col_labels[level]], label)] # type: ignore

    def get_variants(self, level: Level, mode: Mode) -> list[tuple[DataFrame, str]]:
        return [(self.get_variant(level, mode, label), label) for label in self._dataset[self.col_labels[level]].unique()]

    def get_all_variants(self) -> list[tuple[list[tuple[DataFrame, str]], tuple[Level, Mode]]]:
        return [(self.get_variants(l, m), (l,m)) for l in Level for m in Mode]
