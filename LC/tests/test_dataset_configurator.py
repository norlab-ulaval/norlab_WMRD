# coding=utf-8

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass

from LC import data_containers as dcu
from LC import dataset_configurator as dc


@pytest.fixture(scope="function")
def setup_dataset() -> pd.DataFrame:
    return pd.read_pickle('../../data/marmotte/ga_hard_snow_25_01_a/slip_dataset_all.pkl')


def test_extract_dataframe_feature_working(setup_dataset):
    fn = 'body_vel_disturption'
    df = setup_dataset.filter(like=f"{fn}_x")

    container = dc.extract_dataframe_feature(dataset=setup_dataset,
                                             feature_name=fn,
                                             data_container_type=dcu.StatePose)

    assert container.feature_name is fn
    assert container.get_sample_size() == df.shape[0]
    assert container.get_trj_len() == df.shape[1]


def test_extract_dataframe_feature_fail_no_existing_feature(setup_dataset):
    with pytest.raises(ValueError):
        dc.extract_dataframe_feature(dataset=setup_dataset,
                                     feature_name='bodyy_vel_ddisturption',
                                     data_container_type=dcu.StatePose)
