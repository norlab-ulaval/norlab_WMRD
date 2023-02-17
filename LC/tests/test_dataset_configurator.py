# coding=utf-8

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass

from LC import data_containers as dcer
from LC import dataset_configurator as dcor


@pytest.fixture(scope="function")
def setup_dataset() -> pd.DataFrame:
    return pd.read_pickle('../../data/marmotte/ga_hard_snow_25_01_a/slip_dataset_all.pkl')


class TestTimestepIndexingSanityCheck:
    COL_LABEL = 'col_'
    ROW_NB = 10
    COL_NB = 40

    @pytest.fixture(scope="function")
    def setup_dataframe_monotonic_col_label(self):
        def _setup_dataframe_monotonic_col_label(start_index=0):
            return pd.DataFrame(np.ones(
                    (self.ROW_NB, self.COL_NB - start_index)),
                    columns=[self.COL_LABEL + str(i) for i in range(start_index, self.COL_NB)],
                    index=np.arange(start=0, stop=self.ROW_NB))

        return _setup_dataframe_monotonic_col_label

    def test_base_case_pass(self, setup_dataframe_monotonic_col_label):
        tisc = dcor.timestep_indexing_sanity_check(the_dataframe=setup_dataframe_monotonic_col_label(),
                                                   unindexed_column_label=self.COL_LABEL)
        assert type(tisc) is np.ndarray

    def test_index_start_non_zero(self, setup_dataframe_monotonic_col_label):
        with pytest.raises(IndexError):
            tisc = dcor.timestep_indexing_sanity_check(the_dataframe=setup_dataframe_monotonic_col_label(start_index=2),
                                                       unindexed_column_label=self.COL_LABEL)

    def test_missing_step(self, setup_dataframe_monotonic_col_label):
        with pytest.raises(IndexError):
            df_missing = setup_dataframe_monotonic_col_label().drop(f"{self.COL_LABEL}9", axis=1)
            tisc = dcor.timestep_indexing_sanity_check(the_dataframe=df_missing,
                                                       unindexed_column_label=self.COL_LABEL)


class TestExtractDataframeFeature:

    def test_StatePose_working(self, setup_dataset):
        fn = 'body_vel_disturption'
        check_property = 'x'

        container = dcor.extract_dataframe_feature(dataset=setup_dataset, feature_name=fn,
                                                   data_container_type=dcer.StatePose2D)

        df = setup_dataset.filter(like=f"{fn}_{check_property}")
        assert container.feature_name is fn
        assert container.sample_size == df.shape[0]
        assert container.trajectory_len == df.shape[1]

    def test_CmdSkidSteer_working(self, setup_dataset):
        fn = 'cmd'
        check_property = 'left'

        container = dcor.extract_dataframe_feature(dataset=setup_dataset, feature_name=fn,
                                                   data_container_type=dcer.CmdSkidSteer)

        df = setup_dataset.filter(like=f"{fn}_{check_property}")
        assert container.feature_name is fn
        assert container.sample_size == df.shape[0]
        assert container.trajectory_len == df.shape[1]

    def test_bad_argument(self, setup_dataset):
        fn = 'body_vel_disturption'

        mock_value = np.arange(10)
        state_pose = dcer.StatePose2D(feature_name=fn, x=mock_value, y=mock_value, yaw=mock_value,
                                      timestep_index=mock_value)

        with pytest.raises(AttributeError):
            container = dcor.extract_dataframe_feature(dataset=setup_dataset, feature_name=fn,
                                                       data_container_type=state_pose)

    def test_fail_no_existing_feature(self, setup_dataset):
        with pytest.raises(ValueError):
            dcor.extract_dataframe_feature(dataset=setup_dataset, feature_name='bodyy_vel_ddisturption',
                                           data_container_type=dcer.StatePose2D)

    def test_no_existing_feature_dimension(self, setup_dataset):
        with pytest.raises(ValueError):
            dcor.extract_dataframe_feature(dataset=setup_dataset, feature_name='body_vel_disturption',
                                           data_container_type=dcer.CmdStandard)

    def test_missing_timestep(self, setup_dataset):
        col_label = 'body_vel_disturption'
        df_missing = setup_dataset.drop(f"{col_label}_x_9", axis=1)

        with pytest.raises(ValueError):
            dcor.extract_dataframe_feature(dataset=df_missing, feature_name=col_label,
                                           data_container_type=dcer.StatePose2D)
