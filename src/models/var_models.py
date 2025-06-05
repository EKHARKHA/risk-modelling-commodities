import numpy as np
import pandas as pd
from scipy.stats import norm

def var_historical(returns: pd.Series, window: int, alpha: float) -> pd.Series:
    """
    Calcule la VaR historique par sliding window.
    - returns : pd.Series des retours journaliers
    - window : taille de la fenêtre (nombre de jours) pour l'historique
    - alpha : niveau de confiance (ex. 0.95 pour VaR 95%)
    Retourne : pd.Series de VaR indexée par la date (à partir de la fenêtre 'window').
    """
    var_list = []
    dates = returns.index[window:]
    for i in range(window, len(returns)):
        hist_window = returns.iloc[i-window:i]
        var_val = np.percentile(hist_window, 100 * (1 - alpha))
        var_list.append(var_val)
    return pd.Series(var_list, index=dates, name=f"VaR_{int(alpha*100)}")

def var_parametric(returns: pd.Series, window: int, alpha: float) -> pd.Series:
    """
    Calcule la VaR paramétrique (hypothèse normale) par rolling window.
    - returns : pd.Series des retours
    - window : taille de fenêtre
    - alpha : niveau de confiance
    Retourne : pd.Series de VaR indexée par la date.
    """
    var_list = []
    dates = returns.index[window:]
    for i in range(window, len(returns)):
        hist_window = returns.iloc[i-window:i]
        mu = hist_window.mean()
        sigma = hist_window.std(ddof=1)
        var_val = mu + sigma * norm.ppf(1 - alpha)
        var_list.append(var_val)
    return pd.Series(var_list, index=dates, name=f"VaR_param_{int(alpha*100)}")
