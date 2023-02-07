# coding=utf-8

from dataclasses import dataclass, asdict, astuple
import pandas as pd
import numpy as np
from typing import Tuple, NamedTuple, Union, KeysView
from abc import ABC, ABCMeta, abstractmethod


@dataclass()
class Feature(ABC):
    """ Usage: `feature_name` must be the same as in the dataframe column"""
    feature_name: str

    @property
    @abstractmethod
    def _registred_ref_ndarray(self):
        """ Register the top ndarray property for computation of sample size and trajectory len """
        pass

    def get_dimension_names(self) -> KeysView:
        return tuple(asdict(self).keys())

    def get_sample_size(self) -> int:
        return self._registred_ref_ndarray.shape[0]

    def get_trj_len(self) -> int:
        return self._registred_ref_ndarray.shape[1]


@dataclass()
class StatePose(Feature):
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray

    @property
    def _registred_ref_ndarray(self):
        return self.x


@dataclass()
class Cmd(Feature):
    linear_vel: np.ndarray
    angular_vel: np.ndarray

    @property
    def _registred_ref_ndarray(self):
        return self.linear_vel


@dataclass()
class Velocity(Cmd):
    pass


def extract_feature(dataset: pd.DataFrame, feature_name: str, dimensions_name: Tuple[str, ...], trajectory_len: int):
    pass
