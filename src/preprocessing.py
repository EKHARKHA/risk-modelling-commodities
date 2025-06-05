import numpy as np
import pandas as pd

def compute_log_returns(price_series: pd.Series) -> pd.Series:
    """
    Calcule les retours logarithmiques à partir d'une série de prix (Adj Close).
    Ex. : price_series est la colonne 'Adj Close' d'un DataFrame.
    """
    returns = np.log(price_series / price_series.shift(1))
    return returns.dropna()

def train_test_split_ts(returns: pd.Series, test_window: int = 250):
    """
    Sépare une série temporelle de retours en In-Sample (train) et Out-of-Sample (test).
    - returns : pd.Series de retours, indexée par date
    - test_window : nombre d'observations conservées pour le test (ex. : 250 jours)
    Retourne (train_series, test_series).
    """
    train = returns.iloc[:-test_window]
    test = returns.iloc[-test_window:]
    return train, test
