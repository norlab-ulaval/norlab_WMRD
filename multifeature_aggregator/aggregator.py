# coding=utf-8

import pandas as pd
from dataclasses import fields as fields, make_dataclass, dataclass
from multifeature_aggregator.data_containers import (
    AbstractFeatureDataclass, AbstractMultifeature, feature_dataclass_factory, FeatureSpecification, StatePose2D
    )
from typing import Type, Union

from multifeature_aggregator.utils import timestep_indexing_sanity_check


def aggregate_multiple_features(
        dataset_frame: pd.DataFrame, dataset_info: str,
        features_config: dict[str, Union[Type[AbstractFeatureDataclass], tuple[str, ...]]]) -> dataclass:
    """ Extract multiple features from a dataset (formated in a dataframe) based on a configuration dictionary.

    The `features_config` specify the feature name to lookout in the `dataset_frame` header and agregate them in
    a `Multifeature` dataclass. Feature dimensions such as 'x', 'y' 'z' are specified either by using existing
    `AbstractFeatureDataclass` subclass such as:
        `StatePose2D`, `CmdStandard`, `CmdSkidSteer`, `Velocity`, `VelocitySkidSteer`
        or by using tuple of strings such as ('<new feature dataclass type name>', '<dimension names 1>', '<dimension
        names 2>', ...).

        >>> feature_config = {
        >>>             'icp_interpolated': StatePose2D,
        >>>             'idd_vel':          StatePose2D,
        >>>             'icp':              ('StatePose3D', 'x', 'y', 'z', 'roll', 'pitch', 'yaw')
        >>>             }

    :param dataset_frame: Dataset as a panda dataframe
    :param dataset_info: Any relevant information pertaining to the dataset (location, robot, condition ...)
    :param features_config: The features to agregate from the dataset as a configuration dictionary
    """

    features = []
    features_type = []

    for feature_name, feature_dataclass in features_config.items():

        if isinstance(feature_dataclass, tuple):
            if len(feature_dataclass) == 1:
                raise KeyError(
                    f"(!) Check your `features_config` dict. You forgot to specify the '{feature_name}' dimensions.")

            new_type, *dims = feature_dataclass
            feat_spec = FeatureSpecification(new_feature_dataclass_type=new_type, dimension_names=tuple(dims))
            feature_dataclass = feature_dataclass_factory(specification=feat_spec)
            feature = extract_single_feature(dataset=dataset_frame, feature_name=feature_name,
                                             data_container_type=feature_dataclass)
        elif issubclass(feature_dataclass, AbstractFeatureDataclass):
            feature = extract_single_feature(dataset=dataset_frame, feature_name=feature_name,
                                             data_container_type=feature_dataclass)

        features_type.append((feature_name, type(feature)))
        features.append(feature)

    multifeature = make_dataclass('multifeature', bases=(AbstractMultifeature,), fields=features_type)
    return multifeature(dataset_info, *features)


def extract_single_feature(dataset: pd.DataFrame,
                           feature_name: str,
                           data_container_type: Type[AbstractFeatureDataclass]) -> AbstractFeatureDataclass:
    """ Dataframe feature extractor automation function.

    Usage:
     1. Suposing the dataframe head contain column `body_vel_disturption_x_0` to `body_vel_disturption_x_39`.
     2. The `feature_name` and `data_container_type` properties specify which data will be extracted from the
     `dataset_frame`.
     3. `feature_name` must be the same name prefix used in the dataframe head e.g:`body_vel_disturption`.
     4. `data_container.<property name>` must be a postfix to `feature_name_<property name>_<index>` used in the
        dataframe head.

    :param dataset:
    :param feature_name:
    :param data_container_type:

    (NICE TO HAVE) ToDo: implement extract arbitrary trajectory length (ref task SWMRD-12 Implement feature extractor)
    with param: `extract_trajectory_timesteps: Optional[slice] = None`
    """

    try:
        if not issubclass(data_container_type, AbstractFeatureDataclass):
            raise ValueError(f"(!) `{data_container_type}` must be a subclass of `AbstractFeatureDataclass`")
    except TypeError as e:
        raise AttributeError(
                f"(!) `{data_container_type}` must not be instanciated, just pass the class as attribute.")
    else:
        try:
            df_features = dataset.filter(like=feature_name)
            if df_features.empty:
                raise ValueError(
                        f"(!) The parameter `{feature_name}` does not exist in `dataset_frame` as a column header "
                        f"prefix")

            container_properties = fields(data_container_type)[1:]  # Remove 'feature_name'
            tmp_container = { each_field.name: None for each_field in container_properties }

            timestep_index = None
            for each_property in data_container_type.get_dimension_names():
                df_header_field = f"{feature_name}_{each_property}"
                df_property = df_features.filter(regex=f"{df_header_field}_\\d+")
                if df_property.empty:
                    raise ValueError(
                            f"(!) The column `{df_header_field}` does not exist in `dataset_frame`. "
                            f"Check that property `{each_property}` in {str(data_container_type)} is a "
                            f"`{feature_name}` postfix in the dataset_frame")

                try:
                    timestep_index = timestep_indexing_sanity_check(df_property, df_header_field)
                    if tmp_container['timestep_index'] is None:
                        tmp_container['timestep_index'] = timestep_index
                except IndexError as e:
                    raise ValueError(
                            f"(!) There is a problem with the `dataset_frame` column label `{df_header_field}_` "
                            f"timestep "
                            f"index. "
                            f"<< {e}")

                tmp_container[each_property] = df_property.to_numpy()

        except ValueError as e:
            raise

    # noinspection PyArgumentList
    return data_container_type(feature_name=feature_name, **tmp_container)
