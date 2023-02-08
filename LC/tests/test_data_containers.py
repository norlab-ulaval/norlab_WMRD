# coding=utf-8

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, astuple, field, fields

from LC import data_containers as dcu


@dataclass()
class MockDataContainer:
    name: str
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray


@pytest.fixture(scope="class")
def setup_mock_data() -> MockDataContainer:
    return MockDataContainer(name='mock_data',
                             a=np.ones((10, 40)),
                             b=np.ones((10, 40)),
                             c=np.ones((10, 40))
                             )


@pytest.fixture(scope="class")
def setup_mock_data_uneven() -> MockDataContainer:
    return MockDataContainer(name='mock_data_uneven',
                             a=np.ones((10, 40)),
                             b=np.ones((10, 40)),
                             c=np.ones((9, 39))
                             )


class TestFeature:

    @dataclass
    class MockFeatureChild(dcu.FeatureDataclass):
        aa: np.ndarray
        bb: np.ndarray
        cc: np.ndarray

        @property
        def _registred_ref_ndarray(self):
            return self.aa


    @pytest.fixture
    def setup_mock_feature_child(self, setup_mock_data):
        return self.MockFeatureChild(feature_name=setup_mock_data.name,
                                     aa=setup_mock_data.a, bb=setup_mock_data.b, cc=setup_mock_data.c)

    def test_class_feature_post_init_base(self, setup_mock_feature_child, setup_mock_data):
        mfc = setup_mock_feature_child
        assert mfc.aa.size == mfc.bb.size
        assert mfc.bb.size == mfc.cc.size

    def test_class_feature_post_init_check(self, setup_mock_data_uneven):
        with pytest.raises(ValueError):
            mfc_u = self.MockFeatureChild(feature_name=setup_mock_data_uneven.name, aa=setup_mock_data_uneven.a,
                                          bb=setup_mock_data_uneven.b, cc=setup_mock_data_uneven.c)

    def test_get_dimension_names(self, setup_mock_feature_child, setup_mock_data):
        mfc = setup_mock_feature_child
        assert fields(mfc)[0].name not in mfc.get_dimension_names()
        assert mfc.feature_name is setup_mock_data.name
        assert ('aa', 'bb', 'cc') == mfc.get_dimension_names()

    def test_get_dimension_names_on_uninstiated_class(self):
        stp = dcu.StatePose
        stp.get_dimension_names()

    def test_get_sample_size(self, setup_mock_feature_child, setup_mock_data):
        mdc = setup_mock_feature_child
        assert mdc.get_sample_size() == setup_mock_data.a.shape[0]

    def test_get_trj_len(self, setup_mock_feature_child, setup_mock_data):
        mdc = setup_mock_feature_child
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
