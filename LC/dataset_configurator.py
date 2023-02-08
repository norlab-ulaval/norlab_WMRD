# coding=utf-8
from typing import Tuple

import pandas as pd
import numpy as np
from dataclasses import fields as dc_fields
from .data_containers import FeatureDataclass, StatePose, Cmd, Velocity
from typing import Optional


def extract_dataframe_feature(dataset: pd.DataFrame,
                              feature_name: str,
                              data_container_type: FeatureDataclass) -> FeatureDataclass:
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
        df_features = dataset.filter(like=feature_name)
        assert not df_features.empty
    except AssertionError as e:
        raise ValueError(f"(!) The parameter `{feature_name}` does not exist in `dataset` as a column header prefix")

    container_properties = dc_fields(data_container_type)[1:]  # Remove 'feature_name'
    _tmp_container = { each_field.name: None for each_field in container_properties }

    for each_property in data_container_type.get_dimension_names():
        try:
            _df_header_field = f"{feature_name}_{each_property}_"
            df_property = df_features.filter(regex=f"{_df_header_field}\\d+")
            assert not df_property.empty

            _tmp_container[each_property] = df_property.to_numpy()
        except AssertionError as e:
            raise ValueError(
                    f"(!) The column {_df_header_field} does not exist in `dataset`. Check that property `"
                    f"{each_property}` in {str(data_container_type)} is a `{feature_name}` postfix in the dataset"
                    )

    the_data_container_with_extracted_data = data_container_type(feature_name=feature_name, **_tmp_container)
    return the_data_container_with_extracted_data
