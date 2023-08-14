# coding=utf-8
import datetime
from dataclasses import dataclass, asdict, field, fields, make_dataclass
import numpy as np
from typing import Tuple, Type
from abc import ABC
from collections import namedtuple


def extract_class_name_from_type(the_object: object) -> str:
    """ Take an object and the return the class name as a string

    Example:
        >>> aaa = np.ones((2,2))
        >>> extract_class_name_from_type(aaa)
        # "ndarray"

    :param the_object: an instance
    """
    return str(type(the_object)).strip("<'>").split('.')[-1]


@dataclass()
class AbstractFeatureDataclass:
    """ An abstract base dataclass for control related data manipulation

    - (nice to have) ToDo: Add option to use timestamp instead of timestep index
    """
    feature_name: str
    timestep_index: np.ndarray

    @classmethod
    def get_dimension_names(cls) -> Tuple[str, ...]:
        container_properties = fields(cls)[2:]  # Remove the 'feature_name' and `timestep_index` properties
        return tuple(each_field.name for each_field in container_properties)

    @property
    def shape(self) -> tuple[int, int]:
        return self.__getattribute__(self.get_dimension_names()[0]).shape

    @property
    def trajectory_len(self) -> int:
        return len(self.timestep_index)

    @property
    def sample_size(self) -> int:
        return int(self.__getattribute__(self.get_dimension_names()[0]).size / self.trajectory_len)

    def ravel_dimensions_in_place(self) -> None:
        for each_data_property in self.get_dimension_names():
            ravel__copy = self.__getattribute__(each_data_property).ravel()
            self.__setattr__(each_data_property, ravel__copy)
        return None

    def post_init_callback(self, feature_name):
        """ Overide methode to execute custom computation on feature dataclass

        Example:
            >>> @dataclass
            >>> class StatePose2DSteadyState(StatePose2D):
            >>>     def post_init_callback(self, feature_name):
            >>>         feature = self.__getattribute__(feature_name)
            >>>         self.__setattr__(feature_name, feature[steady_state_mask])
            >>>         return None

        """
        pass

    def __str__(self):
        """ User representation. Handle dynamically property added at run time """
        t_sp = " " * 10
        m_sp = " " * 4
        item_space = " " * 3
        dataclass_name = extract_class_name_from_type(self)
        repr_str = f"\n{t_sp}{dataclass_name}(\n"
        v: AbstractFeatureDataclass
        m_sp += t_sp
        for k, v in self.__dict__.items():
            if k == 'feature_name':
                repr_str += f"{m_sp}feature_name: {v}\n"
            elif k == 'timestep_index':
                repr_str += f"{m_sp}sample_size: {self.sample_size}\n"
                repr_str += f"{m_sp}trajectory_len: {self.trajectory_len}\n"
                # repr_str += f"{m_sp}shape: {self.shape}\n"
                repr_str += f"{m_sp}dimensions:\n"
            else:
                repr_str += f"{m_sp}{item_space}{k}: {extract_class_name_from_type(v)} {v.shape}\n"
        repr_str += f"{m_sp})"
        return repr_str

    def __post_init__(self):

        if not self.get_dimension_names():
            raise TypeError(
                    f"(!) AbstractFeatureDataclass is an abstract baseclass, "
                    f"it must be subclassed in order to be instanciated.")

        data_property_shape = None
        for each_name in self.get_dimension_names():
            data_property: np.ndarray = self.__getattribute__(each_name)

            self.post_init_callback(feature_name=each_name)

            if type(data_property) is not np.ndarray:
                raise TypeError(f"(!) Property `{each_name}` is not a numpy ndarray")
            elif data_property_shape is not None:
                if data_property_shape != data_property.shape:
                    raise ValueError(f"(!) `{self.feature_name}` container received numpy array's with uneven shape")

                data_property_shape = data_property.shape
            else:
                data_property_shape = data_property.shape


        return None


@dataclass()
class FeatureSpecification:
    new_feature_dataclass_type: str
    dimension_names: tuple[str, ...]


def feature_dataclass_factory(specification: FeatureSpecification) -> Type[AbstractFeatureDataclass]:
    """ Factory to dynnamicaly create new `AbstractFeatureDataclass` subclass from dictionary specification.

    Usage example:
        >>> spec_ = FeatureSpecification(new_feature_dataclass_type='my_new_feature_type',
        >>>                              dimension_names=('xx', 'yy', 'yawww'))
        >>> the_new_cls = feature_dataclass_factory(specification=spec_)
        >>> assert issubclass(the_new_cls, AbstractFeatureDataclass)
        >>> # True
        >>> assert isinstance(the_new_cls, AbstractFeatureDataclass)
        >>> # False

    :param specification: A FeatureSpecification object,
    :return: A new subclass of AbstractFeatureDataclass
    """

    # (NICE TO HAVE) ToDo: the following try/except logic can be refactored out to the FeatureSpecification dataclass
    try:
        new_feature_dataclass_type: str = specification.new_feature_dataclass_type
        dimension_names: Tuple[str, ...] = specification.dimension_names
        if (type(new_feature_dataclass_type) is not str) or (type(dimension_names) is not tuple) or (
                any([type(name) is not str for name in dimension_names])):
            raise AttributeError
    except KeyError as e:
        raise KeyError(
                f"(!) The key {e} is missing from the `specification` dictionary passed in argument.")
    except AttributeError as e:
        raise AttributeError(
                f"(!) The `specification` dictionary must contain key 'new_feature_dataclass_type' with value of type "
                f"string and key `dimension_names` with value of type tuple of string. {e}")

    properties_field = [(name, np.ndarray) for name in dimension_names]

    return make_dataclass(new_feature_dataclass_type, bases=(AbstractFeatureDataclass,), fields=properties_field)


@dataclass()
class StatePose2D(AbstractFeatureDataclass):
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
    state_pose_k: StatePose2D
    velocity_skid_steer_k_previous: VelocitySkidSteer
    cmd_skid_steer_k: CmdSkidSteer
    cmd_skid_steer_k_previous: CmdSkidSteer


@dataclass
class AbstractMultifeature:
    dataset_info: str
    aggregated_date: datetime.datetime = field(init=False)

    def __post_init__(self):
        self.aggregated_date = datetime.datetime.now()

    def __str__(self):
        """ User representation. Handle dynamically property added at run time """
        t_sp = " " * 0
        m_sp = " " * 3
        repr_str = f"\n{t_sp}Multifeature(\n"
        m_sp += t_sp
        for k, v in self.__dict__.items():
            if k == 'dataset_info':
                repr_str += f"{m_sp}dataset_info: {v}\n"
                repr_str += f"{m_sp}aggregated_date: {self.aggregated_date}\n"
            elif k == 'aggregated_date':
                pass
            else:
                # repr_str += f"{m_sp}{k}( {str(v)}\n{m_sp*2})\n"
                repr_str += f"{m_sp}{k}: {str(v)}\n"
        repr_str += f"{m_sp})"
        return repr_str

    @property
    def summary(self) -> None:
        print(self)
        return None