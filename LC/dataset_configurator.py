# coding=utf-8

import pandas as pd
import numpy as np
from dataclasses import fields as dc_fields
from .data_containers import AbstractFeatureDataclass, StatePose, CmdStandard, Velocity
from typing import List, Callable, Type, Any


def timestep_indexing_sanity_check(the_dataframe: pd.DataFrame, unindexed_column_label: str) -> np.ndarray:
    """ Utility for validating timestep index in column label by checking if it is either missing a step or not
    monotonicaly increassing.

    :param the_dataframe:
    :param unindexed_column_label:
    :return: the timestep index as an numpy ndarray or raise a IndexError
    """
    column_labels: List[str] = the_dataframe.columns.to_list()
    col_index = [int(each_label.strip(unindexed_column_label)) for each_label in column_labels]
    timestep_index = np.array(col_index)
    np_diff = np.diff(timestep_index)

    index_is_monoticaly_increasing = np.all(np_diff > 0)

    column_nb = the_dataframe.shape[1]
    index_start = timestep_index[0]
    if index_start == 0:
        index_start = -1
    index_end = timestep_index[-1]
    delta = index_end - index_start
    index_has_constant_increment = column_nb == delta

    is_timestep_index_good_to_go = all((index_is_monoticaly_increasing, index_has_constant_increment))
    if not is_timestep_index_good_to_go:
        raise IndexError(f"(!) The timestep index is either missing a step or not monotonicaly increassing")

    return timestep_index


def extract_dataframe_feature(dataset: pd.DataFrame,
                              feature_name: str,
                              data_container_type: Type[AbstractFeatureDataclass]) -> AbstractFeatureDataclass:
    """ Dataframe feature extractor automation function.

    Usage:
     1. Suposing the dataframe head contain column 'body_vel_disturption_x_0' to `body_vel_disturption_x_39`.
     2. The `feature_name` and `data_container_type` properties specify which data will be extracted from the `dataset`.
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
                        f"(!) The parameter `{feature_name}` does not exist in `dataset` as a column header prefix")

            container_properties = dc_fields(data_container_type)[1:]  # Remove 'feature_name'
            tmp_container = { each_field.name: None for each_field in container_properties }

            timestep_index = None
            for each_property in data_container_type.get_dimension_names():
                df_header_field = f"{feature_name}_{each_property}"
                df_property = df_features.filter(regex=f"{df_header_field}_\\d+")
                if df_property.empty:
                    raise ValueError(
                            f"(!) The column `{df_header_field}` does not exist in `dataset`. "
                            f"Check that property `{each_property}` in {str(data_container_type)} is a "
                            f"`{feature_name}` postfix in the dataset")

                try:
                    timestep_index = timestep_indexing_sanity_check(df_property, df_header_field)
                    if tmp_container['timestep_index'] is None:
                        tmp_container['timestep_index'] = timestep_index
                except IndexError as e:
                    raise ValueError(
                            f"(!) There is a problem with the `dataset` column label `{df_header_field}_` timestep "
                            f"index. "
                            f"<< {e}")

                tmp_container[each_property] = df_property.to_numpy()

        except ValueError as e:
            raise

    # noinspection PyArgumentList
    return data_container_type(feature_name=feature_name, **tmp_container)
