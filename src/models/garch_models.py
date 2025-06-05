from arch import arch_model
import numpy as np
import pandas as pd
from scipy.stats import t, norm

def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "normal"):
    """
    Calibre un modèle GARCH(p,q) sur la série 'returns' (rendements en décimal, ex. 0.001).
    - returns : pd.Series des retours journaliers
    - p, q : ordres GARCH
    - dist : "normal" ou "t" (Student-t)
    Retourne l'objet résultat du fit (ARCHModelResult).
    """
    # On multiplie par 100 car arch_model attend des données en pourcentage (100*0.001 = 0.1%)
    am = arch_model(returns * 100, vol="GARCH", p=p, q=q, dist=dist, rescale=False)
    res = am.fit(disp="off")
    return res

def forecast_garch_var(fitted_res, start_dates: pd.DatetimeIndex, returns: pd.Series, alpha: float):
    """
    Calcule la VaR conditionnelle jour par jour, à partir d'un modèle GARCH calibré.
    - fitted_res : résultat de fit_garch (objet ARCHModelResult)
    - start_dates : pd.DatetimeIndex (dates Out-of-Sample)
    - returns : pd.Series complète (In-Sample + Out-of-Sample)
    - alpha : niveau de confiance (ex. 0.95 pour VaR 95%)
    Retourne : pd.Series indexée par start_dates, contenant les valeurs de VaR conditionnelle.
    """
    var_list = []

    # Récupérer la "vraie" distribution utilisée lors du fit in-sample
    # fitted_res.model.distribution.name renvoie typiquement "Normal" ou "StudentsT"
    dist_name_raw = fitted_res.model.distribution.name.lower()

    # Normaliser : si la chaîne contient "t", on retient "t", sinon "normal"
    if "t" in dist_name_raw:
        dist_model = "t"
    else:
        dist_model = "normal"

    for date in start_dates:
        # 1) Trouver l'emplacement de la date dans la série complète
        end_loc = returns.index.get_loc(date)

        # 2) Extraire les retours jusqu'au jour précédent (in-sample pour ce date)
        in_sample_returns = returns.iloc[:end_loc]

        # 3) Recalibrer un modèle GARCH(1,1) sur la partie in-sample
        model = arch_model(
            in_sample_returns * 100,
            vol="GARCH", p=1, q=1,
            dist=dist_model,
            rescale=False
        )
        res = model.fit(disp="off")

        # 4) Estimer la variance 1 jour à l’avance
        forecast = res.forecast(horizon=1, reindex=False)
        sigma_t = np.sqrt(forecast.variance.values[-1, 0]) / 100  # retourner en décimal
        mu_t = forecast.mean.values[-1, 0] / 100

        # 5) Choisir le quantile selon la distribution
        if dist_model == "normal":
            q = norm.ppf(1 - alpha)
        else:
            # Pour Student-t, on récupère nu (degrés de liberté)
            nu = res.params.get("nu", None)
            if nu is None:
                # Par sécurité, si jamais "nu" manque, on retombe sur une normale
                q = norm.ppf(1 - alpha)
            else:
                q = t.ppf(1 - alpha, df=nu)

        # 6) Calculer la VaR conditionnelle
        var_val = mu_t + sigma_t * q
        var_list.append(var_val)

    return pd.Series(var_list, index=start_dates, name=f"VaR_GARCH_{int(alpha*100)}")


