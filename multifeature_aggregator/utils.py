# coding=utf-8
import numpy as np
import pandas as pd


def timestep_indexing_sanity_check(the_dataframe: pd.DataFrame, unindexed_column_label: str) -> np.ndarray:
    """ Utility for validating timestep index in column label by checking if it is either missing a step or not
    monotonicaly increassing.

    :param the_dataframe:
    :param unindexed_column_label:
    :return: the timestep index as a numpy ndarray or raise a IndexError
    """
    column_labels: list[str] = the_dataframe.columns.to_list()
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
