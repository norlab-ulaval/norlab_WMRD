# coding=utf-8

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, astuple, field, fields

from LC import data_containers as dcu



@dataclass()
class MockData:
    name: str
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray


@pytest.fixture(scope="class")
def setup_mock_data() -> MockData:
    return MockData(name='mock_data',
                    a=np.ones((10, 40)),
                    b=np.ones((10, 40)),
                    c=np.ones((10, 40))
                    )


class TestFeature:

    @dataclass
    class MockDataclass(dcu.FeatureDataclass):
        x: np.ndarray

        @property
        def _registred_ref_ndarray(self):
            return self.x


    @pytest.fixture
    def setup_mock_dataclass(self, setup_mock_data):
        return self.MockDataclass(feature_name=setup_mock_data.name, x=setup_mock_data.a)

    def test_class_feature_init(self, setup_mock_dataclass):
        assert type(setup_mock_dataclass.x) is np.ndarray

    def test_get_dimension_names(self, setup_mock_dataclass, setup_mock_data):
        mdc = setup_mock_dataclass
        assert 'feature_name' not in mdc.get_dimension_names()
        assert mdc.feature_name is setup_mock_data.name
        assert 'x' in mdc.get_dimension_names()

    def test_get_dimension_names_on_uninstiated_class(self):
        stp = dcu.StatePose
        stp.get_dimension_names()

    def test_get_sample_size(self, setup_mock_dataclass, setup_mock_data):
        mdc = setup_mock_dataclass
        assert mdc.get_sample_size() == setup_mock_data.a.shape[0]

    def test_get_trj_len(self, setup_mock_dataclass, setup_mock_data):
        mdc = setup_mock_dataclass
        assert mdc.get_trj_len() == setup_mock_data.a.shape[1]


class TestStatePose:

    def test_init_empty_data_properties(self, setup_mock_data):
        md = setup_mock_data
        with pytest.raises(TypeError):
            sp = dcu.StatePose(feature_name=md.name)

    def test_init(self, setup_mock_data):
        md = setup_mock_data
        sp = dcu.StatePose(feature_name=md.name, x=md.a, y=md.b, yaw=md.c)


class TestCmdAndVelocity:

    def test_Cmd_init(self, setup_mock_data):
        md = setup_mock_data
        cmd = dcu.Cmd(feature_name=md.name, linear_vel=md.a, angular_vel=md.b)

    def test_Velocity_init(self, setup_mock_data):
        md = setup_mock_data
        vel = dcu.Velocity(feature_name=md.name, linear_vel=md.a, angular_vel=md.b)

        assert isinstance(vel, dcu.Velocity)
