# coding=utf-8

import pytest
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, astuple, field, fields
from typing import Type

from LC import data_containers as dcer


@dataclass()
class MockDataContainer:
    name: str
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    ts: np.ndarray


@pytest.fixture(scope="class")
def setup_mock_data() -> MockDataContainer:
    return MockDataContainer(name='mock_data',
                             a=np.ones((10, 40)),
                             b=np.ones((10, 40)),
                             c=np.ones((10, 40)),
                             ts=np.arange(0, 40)
                             )


@pytest.fixture(scope="class")
def setup_mock_data_uneven() -> MockDataContainer:
    return MockDataContainer(name='mock_data_uneven',
                             a=np.ones((10, 40)),
                             b=np.ones((10, 40)),
                             c=np.ones((9, 39)),
                             ts=np.arange(0, 40)
                             )


class TestFeature:

    @dataclass
    class MockFeatureChild(dcer.AbstractFeatureDataclass):
        aa: np.ndarray
        bb: np.ndarray
        cc: np.ndarray


    @pytest.fixture
    def setup_mock_feature_child(self, setup_mock_data):
        return self.MockFeatureChild(feature_name=setup_mock_data.name,
                                     aa=setup_mock_data.a, bb=setup_mock_data.b, cc=setup_mock_data.c,
                                     timestep_index=setup_mock_data.ts)

    def test_FeatureDataclass_baseclass_not_instantiable(self):
        with pytest.raises(TypeError):
            shouldfail = dcer.AbstractFeatureDataclass(feature_name='try_to_do_it')

    def test_class_feature_post_init_base(self, setup_mock_feature_child, setup_mock_data):
        mfc = setup_mock_feature_child
        assert mfc.aa.size == mfc.bb.size
        assert mfc.bb.size == mfc.cc.size

    def test_class_feature_post_init_check(self, setup_mock_data_uneven):
        with pytest.raises(ValueError):
            mfc_u = self.MockFeatureChild(feature_name=setup_mock_data_uneven.name, aa=setup_mock_data_uneven.a,
                                          bb=setup_mock_data_uneven.b, cc=setup_mock_data_uneven.c,
                                          timestep_index=setup_mock_data_uneven.ts)

    def test_get_dimension_names(self, setup_mock_feature_child, setup_mock_data):
        mfc = setup_mock_feature_child
        for each in ('feature_name', 'timestep_index'):
            assert hasattr(mfc, each)
            assert each not in mfc.get_dimension_names()
        assert mfc.feature_name is setup_mock_data.name
        assert ('aa', 'bb', 'cc') == mfc.get_dimension_names()

    def test_get_dimension_names_on_uninstiated_class(self):
        stp = dcer.StatePose
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
            sp = dcer.StatePose(feature_name=md.name)

    def test_init(self, setup_mock_data):
        md = setup_mock_data
        sp = dcer.StatePose(feature_name=md.name, x=md.a, y=md.b, yaw=md.c, timestep_index=md.ts)


class TestCmdAndVelocity:

    def test_Cmd_init(self, setup_mock_data):
        md = setup_mock_data
        cmd = dcer.CmdStandard(feature_name=md.name, linear_vel=md.a, angular_vel=md.b, timestep_index=md.ts)

    def test_Velocity_init(self, setup_mock_data):
        md = setup_mock_data
        vel = dcer.Velocity(feature_name=md.name, linear_vel=md.a, angular_vel=md.b, timestep_index=md.ts)

        assert isinstance(vel, dcer.Velocity)


class TestfeatureDataclassFactory:

    @pytest.fixture
    def setup_config(self):
        spec = dict(feature_name='mock_data', dimension_names=['xx', 'yy', 'yaww'])
        return spec

    def test_spec_ok(self, setup_config):
        dcer.feature_dataclass_factory(specification=setup_config)

    def test_bad_spec(self):
        with pytest.raises(KeyError):
            fdf1 = dcer.feature_dataclass_factory(
                    specification=dict(feature_AAAA='mock_data', dimension_names=['xx', 'yy', 'yaww']))
            fdf2 = dcer.feature_dataclass_factory(
                    specification=dict(feature_name='mock_data', dimension_AAAA=['xx', 'yy', 'yaww']))
        with pytest.raises(AttributeError):
            fdf3 = dcer.feature_dataclass_factory(
                    specification=dict(feature_name='mock_data', dimension_names=['xx', 'yy', 999]))

    def test_output_ok(self, setup_config):
        mock_cls = dcer.feature_dataclass_factory(specification=setup_config)
        mock_value = np.arange(10)
        assert issubclass(mock_cls, dcer.AbstractFeatureDataclass)
        assert not isinstance(mock_cls, dcer.AbstractFeatureDataclass)
        mock_cls_instance = mock_cls(feature_name='mock_data',
                                     xx=mock_value, yy=mock_value, yaww=mock_value,
                                     timestep_index=mock_value)
        assert isinstance(mock_cls_instance, dcer.AbstractFeatureDataclass)
        assert mock_cls.get_dimension_names() == ('xx', 'yy', 'yaww')
        assert hasattr(mock_cls_instance, 'feature_name')
        assert hasattr(mock_cls_instance, 'xx')
        assert hasattr(mock_cls_instance, 'yy')
        assert hasattr(mock_cls_instance, 'yaww')

        print('\n'*2, mock_cls_instance)
