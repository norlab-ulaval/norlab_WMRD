# coding=utf-8

from dataclasses import dataclass, asdict, field, fields
import numpy as np
from typing import KeysView, Tuple
from abc import ABC, abstractmethod


@dataclass()
class AbstractFeatureDataclass(ABC):
    """ An abstract base dataclass for control related data manipulation """
    feature_name: str

    @classmethod
    def get_dimension_names(cls) -> Tuple[str, ...]:
        container_properties = fields(cls)[1:]  # Remove the 'feature_name' property
        return tuple(each_field.name for each_field in container_properties)

    def get_sample_size(self) -> int:
        return self.__getattribute__(self.get_dimension_names()[0]).shape[0]

    def get_trj_len(self) -> int:
        return self.__getattribute__(self.get_dimension_names()[0]).shape[1]

    def __post_init__(self):

        if len(self.get_dimension_names()) == 0:
            raise TypeError(
                    f"(!) AbstractFeatureDataclass is an abstract baseclass, "
                    f"it must be subclassed in order to be instanciated.")

        data_property_shape = None

        for each_name in self.get_dimension_names():
            data_property: np.ndarray = self.__getattribute__(each_name)

            if type(data_property) is not np.ndarray:
                raise TypeError(f"(!) Property `{each_name}` is not a numpy ndarray")
            elif data_property_shape is not None:
                if data_property_shape != data_property.shape:
                    raise ValueError(f"(!) `{self.feature_name}` container received numpy array's with uneven shape")
                else:
                    data_property_shape = data_property.shape
            else:
                data_property_shape = data_property.shape
        return None


@dataclass()
class StatePose(AbstractFeatureDataclass):
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


@dataclass()
class Cmd(AbstractFeatureDataclass):
    linear_vel: np.ndarray
    angular_vel: np.ndarray


@dataclass()
class Velocity(Cmd):
    pass
