# -*- coding: utf-8 -*-
"""
Econ 406

unit tests for each of the 4 functions
"""
import matplotlib.pyplot as plt
import src.functions as functions

# test for import_and_clean_data function
def test_import_and_clean_data_has_no_null():
    """Test that output dataframe doesn't contain null values"""
    assert functions.import_and_clean_data().isnull().sum().sum() == 0

# test for data_visualization function
def test_data_visualization_output_two_plots():
    """Test that the number of plots output is two"""
    functions.data_visualization()
    assert plt.gcf().number == 2

# tests for generate_stats function
def test_generate_stats_num_col():
    """Test that the number of columns the table generate is eight"""
    table = functions.generate_stats()
    assert len(table) == 8

def test_generate_stats_has_no_null():
    """Test that output dataframe doesn't contain null values"""
    assert functions.generate_stats().isnull().sum().sum() == 0

# test for run_model function
def test_run_model_output_plots():
    """Test that output more plots after running this function"""
    num_plot_before = plt.gcf().number
    functions.run_model()
    num_plot_after = plt.gcf().number
    assert num_plot_before < num_plot_after
        