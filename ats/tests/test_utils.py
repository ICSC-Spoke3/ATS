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

    def test_normalizza_parametro_vario(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        normalized = normalizza_parametro(df, "a")
        # min-max -> (x - 1) / (5 - 1)
        expected = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0], name="a")
        #È una funzione di pandas che confronta due pd.Series e lancia un errore se non sono identiche
        pd.testing.assert_series_equal(normalized.reset_index(drop=True), expected)

    def test_normalizza_parametro_costante(self):
        df = pd.DataFrame({"b": [7, 7, 7]})
        normalized = normalizza_parametro(df, "b")
        expected = pd.Series([1, 1, 1], name="b")
        pd.testing.assert_series_equal(normalized.reset_index(drop=True), expected)
