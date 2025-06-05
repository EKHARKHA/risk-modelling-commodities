import yfinance as yf
import pandas as pd

def download_commodity(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge l'historique de prix ajustés d'une commodity via yfinance.
    - ticker : ex. "GC=F" pour l'or, "CL=F" pour le pétrole
    - start, end : chaînes au format "YYYY-MM-DD"
    Renvoie un DataFrame avec les colonnes ['Open','High','Low','Close','Adj Close','Volume'] indexé par la date.
    """
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.index.name = 'Date'
    return df

def save_csv(dataframe: pd.DataFrame, file_path: str):
    """
    Sauvegarde un DataFrame en CSV à l'emplacement spécifié.
    """
    dataframe.to_csv(file_path)

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Charge un DataFrame depuis un fichier CSV (parse dates).
    """
    return pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
