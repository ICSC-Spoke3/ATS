import unittest
import pandas as pd
import plotly.graph_objects as go

from ..utils import generate_timeseries_df, normalize_parameter, normalize_df, plot_3d_interactive

# Setup logging
from .. import logger
logger.setup()


class TestUtils(unittest.TestCase):

    def test_generate_timeseries_df(self):

        timeseries_df = generate_timeseries_df(entries=10, variables=2)
        self.assertEqual(timeseries_df.shape, (10,2))

    # Tests for normalize_parameter
    def test_normalize_generic_parameter(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        normalized = normalize_parameter(df, "a")
        expected = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], name="a")
        #È una funzione di pandas che confronta due pd.Series e lancia un errore se non sono identiche
        pd.testing.assert_series_equal(normalized.reset_index(drop=True), expected)

    def test_normalize_costant_parameter(self):
        df = pd.DataFrame({"b": [7, 7, 7]})
        normalized = normalize_parameter(df, "b")
        expected = pd.Series([1, 1, 1], name="b")
        pd.testing.assert_series_equal(normalized.reset_index(drop=True), expected)
    
    # Tests for normalize_df
    def test_normalize_all_columns(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })
        df_norm = normalize_df(df)
        expected = pd.DataFrame({
            "a_norm": [0.0, 0.25, 0.5, 0.75, 1.0],
            "b_norm": [0.0, 0.25, 0.5, 0.75, 1.0]
        })
        pd.testing.assert_frame_equal(df_norm, expected)
    
    def test_normalize_subset_of_columns(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [10, 10, 10],
            "context": [True, False, True],
            "name": ["x", "y", "z"]
        })
        df_norm = normalize_df(df,parameters_subset=["a"])
        expected = pd.DataFrame({
            "a_norm": [0.0, 0.5, 1],
        })
        pd.testing.assert_frame_equal(df_norm, expected)

        df_norm = normalize_df(df,parameters_subset=["a","context"])
        pd.testing.assert_frame_equal(df_norm, expected)

        df_norm = normalize_df(df,parameters_subset=["a","context","name"])
        pd.testing.assert_frame_equal(df_norm, expected)

    # Test for plot_3d_interactive
    def setUp(self):
        self.df = pd.DataFrame({
            "avg_err": [1, 2, 3, 4, 5],
            "max_err": [2, 4, 6, 8, 10],
            "ks_pvalue": [0.1, 0.2, 0.3, 0.4, 0.5],
            "fitness": [7, 8, 10, 18, 13],
            "extra": [10, 20, 30, 40, 50]
        })
    def test_returns_figure_andfilter(self):
        filters = {"max_err": (None, 5)}
        
        fig = plot_3d_interactive(self.df, renderer="json",show = False,filters=filters)  
        self.assertIsInstance(fig, go.Figure)
        y_values_filtered = fig.data[0].y
        self.assertTrue(all(val <= 5 for val in y_values_filtered))

        filters = {"max_err": (2, 8)}
        fig = plot_3d_interactive(self.df, renderer="json",show = False,filters=filters)  
        y_values_filtered = fig.data[0].y
        self.assertTrue(all(2 <= val <= 8 for val in y_values_filtered))

    
