# coding=utf-8

from dataclasses import dataclass, asdict, fields, make_dataclass
import numpy as np
from typing import Tuple, Type
from abc import ABC


@dataclass()
class AbstractFeatureDataclass(ABC):
    """ An abstract base dataclass for control related data manipulation

    - (nice to have) ToDo: Add option to use timestamp instead of timestep index
    """
    feature_name: str
    timestep_index: np.ndarray

    @classmethod
    def get_dimension_names(cls) -> Tuple[str, ...]:
        container_properties = fields(cls)[2:]  # Remove the 'feature_name' and `timestep_index` properties
        return tuple(each_field.name for each_field in container_properties)

    def get_sample_size(self) -> int:
        return self.__getattribute__(self.get_dimension_names()[0]).shape[0]

    def get_trj_len(self) -> int:
        return self.__getattribute__(self.get_dimension_names()[0]).shape[1]

    def post_init_callback(self):
        """ Overide methode to execute custom computation on feature dataclass """
        pass

    def __post_init__(self):

        if not self.get_dimension_names():
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

                data_property_shape = data_property.shape
            else:
                data_property_shape = data_property.shape

            self.post_init_callback()

        return None


def feature_dataclass_factory(specification: dict[str, list[str, ...]]) -> Type[AbstractFeatureDataclass]:
    """ Factory to dynnamicaly create new `AbstractFeatureDataclass` subclass from dictionary specification.

    Usage example:

    >>> spec_ = dict(feature_name='mock_data', dimension_names=['xx', 'yy', 'yaww'])
    >>> new_cls = feature_dataclass_factory(specification=spec_)
    >>> assert issubclass(new_cls, AbstractFeatureDataclass)
    >>> # True
    >>> assert isinstance(new_cls, AbstractFeatureDataclass)
    >>> # False

    :param specification: A dictionary with keys:value 'feature_name': str, 'dimension_names': list[str, ...]}
    :return: A new subclass of AbstractFeatureDataclass
    """
    try:
        feature_name: str = specification['feature_name']
        dimension_names: Tuple[str, ...] = specification['dimension_names']
        if (type(feature_name) is not str) or (type(dimension_names) is not list) or (
                any([type(name) is not str for name in dimension_names])):
            raise AttributeError
    except KeyError as e:
        raise KeyError(
                f"(!) The key {e} is missing from the `specification` dictionary passed in argument.")
    except AttributeError as e:
        raise AttributeError(
                f"(!) The `specification` dictionary must contain key 'feature_name' with value of type string and key "
                f"`dimension_names` with value of type list of string. {e}")

    properties_field = [(name, np.ndarray) for name in dimension_names]

    return make_dataclass(feature_name, bases=(AbstractFeatureDataclass,), fields=properties_field)


@dataclass()
class StatePose(AbstractFeatureDataclass):
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


@dataclass()
class CmdStandard(AbstractFeatureDataclass):
    linear_vel: np.ndarray
    angular_vel: np.ndarray


@dataclass()
class CmdSkidSteer(AbstractFeatureDataclass):
    left: np.ndarray
    right: np.ndarray


@dataclass()
class Velocity(CmdStandard):
    pass


@dataclass()
class VelocitySkidSteer(CmdSkidSteer):
    pass


@dataclass()
class DisturbanceQueryState:
    state_pose_k: StatePose
    velocity_skid_steer_k_previous: VelocitySkidSteer
    cmd_skid_steer_k: CmdSkidSteer
    cmd_skid_steer_k_previous: CmdSkidSteer
