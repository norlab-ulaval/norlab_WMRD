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


class TestTimestepIndexingSanityCheck:
    col_label = 'col_'
    row_nb = 10
    col_nb = 40

    @pytest.fixture(scope="function")
    def setup_dataframe_monotonic_col_label(self):
        def _setup_dataframe_monotonic_col_label(start_index=0):
            return pd.DataFrame(np.ones(
                    (self.row_nb, self.col_nb - start_index)),
                    columns=[self.col_label + str(i) for i in range(start_index, self.col_nb)],
                    index=np.arange(start=0, stop=self.row_nb))

        return _setup_dataframe_monotonic_col_label

    def test_base_case_pass(self, setup_dataframe_monotonic_col_label):
        tisc = dc.timestep_indexing_sanity_check(the_dataframe=setup_dataframe_monotonic_col_label(),
                                                 unindexed_column_label=self.col_label)
        assert tisc is True

    def test_index_start_non_zero(self, setup_dataframe_monotonic_col_label):
        with pytest.raises(IndexError):
            tisc = dc.timestep_indexing_sanity_check(the_dataframe=setup_dataframe_monotonic_col_label(start_index=2),
                                                     unindexed_column_label=self.col_label)
            assert tisc is True


class TestExtractDataframeFeature:

    def test_working(self, setup_dataset):
        fn = 'body_vel_disturption'

        container = dc.extract_dataframe_feature(dataset=setup_dataset, feature_name=fn,
                                                 data_container_type=dcu.StatePose)

        df = setup_dataset.filter(like=f"{fn}_x")
        assert container.feature_name is fn
        assert container.get_sample_size() == df.shape[0]
        assert container.get_trj_len() == df.shape[1]

    def test_bad_argument(self, setup_dataset):
        fn = 'body_vel_disturption'

        mock_value = np.arange(10)
        state_pose = dcu.StatePose(feature_name=fn, x=mock_value, y=mock_value, yaw=mock_value)

        with pytest.raises(AttributeError):
            container = dc.extract_dataframe_feature(dataset=setup_dataset, feature_name=fn,
                                                     data_container_type=state_pose)

    def test_fail_no_existing_feature(self, setup_dataset):
        with pytest.raises(ValueError):
            dc.extract_dataframe_feature(dataset=setup_dataset, feature_name='bodyy_vel_ddisturption',
                                         data_container_type=dcu.StatePose)

    def test_no_existing_feature_dimension(self, setup_dataset):
        with pytest.raises(ValueError):
            dc.extract_dataframe_feature(dataset=setup_dataset, feature_name='body_vel_disturption',
                                         data_container_type=dcu.Cmd)

    def test_missing_timestep(self, setup_dataset):
        col_label = 'body_vel_disturption'
        df_missing = setup_dataset.drop(f"{col_label}_x_9", axis=1)

        with pytest.raises(ValueError):
            dc.extract_dataframe_feature(dataset=df_missing, feature_name=col_label,
                                         data_container_type=dcu.StatePose)
