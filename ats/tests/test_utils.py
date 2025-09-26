import unittest
import pandas as pd

from ..utils import generate_timeseries_df, normalizza_parametro, normalizzazione_df

# Setup logging
from .. import logger
logger.setup()


class TestUtils(unittest.TestCase):

    def test_generate_timeseries_df(self):

        timeseries_df = generate_timeseries_df(entries=10, variables=2)
        self.assertEqual(timeseries_df.shape, (10,2))

    # Tests for normalizza_parametro
    def test_normalizza_parametro_vario(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        normalized = normalizza_parametro(df, "a")
        expected = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], name="a")
        #È una funzione di pandas che confronta due pd.Series e lancia un errore se non sono identiche
        pd.testing.assert_series_equal(normalized.reset_index(drop=True), expected)

    def test_normalizza_parametro_costante(self):
        df = pd.DataFrame({"b": [7, 7, 7]})
        normalized = normalizza_parametro(df, "b")
        expected = pd.Series([1, 1, 1], name="b")
        pd.testing.assert_series_equal(normalized.reset_index(drop=True), expected)
    
    # Tests for normalizzazione_df
    def test_normalize_all_columns(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50]
        })
        df_norm = normalizzazione_df(df)
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
        df_norm = normalizzazione_df(df,parameters_subset=["a"])
        expected = pd.DataFrame({
            "a_norm": [0.0, 0.5, 1],
        })
        pd.testing.assert_frame_equal(df_norm, expected)

        df_norm = normalizzazione_df(df,parameters_subset=["a","context"])
        pd.testing.assert_frame_equal(df_norm, expected)

        df_norm = normalizzazione_df(df,parameters_subset=["a","context","name"])
        pd.testing.assert_frame_equal(df_norm, expected)

