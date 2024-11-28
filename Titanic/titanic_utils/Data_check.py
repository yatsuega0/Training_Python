import pandas as pd
import os

# Examples
# import sys
# sys.path.append('../titanic_utils')  titanic_utilsが存在するパスを指定
# import data_check as ch
# ch.xxxx()

# data_loader.py
def load_titanic_data(filename, directory='data', **kwargs):
    """
    指定されたファイル名でタイタニックデータセットを読み込む関数。

    Parameters
    ----------
    filename : str
        ディレクトリにあるファイルの名前。
    directory : str, optional
        データセットが保存されているディレクトリのパス。デフォルトは 'titanic'。
    **kwargs : dict
        pd.read_csv()へ渡す追加のキーワード引数。

    Returns
    -------
    DataFrame
        読み込まれたデータを含むDataFrame。

    Examples
    --------
    # train.csvをデフォルトのディレクトリから読み込む
    train_data = load_titanic_data('train.csv')

    # test.csvをデフォルトのディレクトリから読み込む、欠損値を'NA'で指定
    test_data = load_titanic_data('test.csv', na_values=['NA'])
    """
    path = f"{directory}/{filename}"

    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {filename} does not exist in the directory {directory}.")

    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except pd.errors.ParserError:
        raise ValueError("Error parsing the file.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


# data summary
def summarize_dataframe(df: pd.DataFrame):
    """
    指定されたデータフレームについて、その形状、カラム、データ型、各カラムの欠損値の数を出力する関数。

    Parameters
    ----------
    df : pandas.DataFrame
        要約するDataFrame。

    Returns
    -------
    None
        要約を直接出力

    Examples
    --------
    summarize_dataframe(df)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('引数 df は pandas.DataFrame インスタンスでなければなりません。')

    print("データフレームのサイズ:", df.shape)
    print("カラムの一覧:")
    print(df.columns)
    print("\n各カラムのデータ型:")
    print(df.dtypes)
    print("\nカラムごとの欠損値の数:")
    print(df.isnull().sum())


# Calculate survival rate
def calculate_survival_rate(data, category):
    """
    指定されたカテゴリに基づいて生存率を計算する関数。

    Parameters
    ----------
    data : DataFrame
        生存データを含むDataFrame。
    category : str
        生存率を計算するカテゴリ（例：'Sex'、'Pclass'）

    Returns
    -------
    dict
        カテゴリごとの生存率を含む辞書
    Examples
    --------
    survival_rates = calculate_survival_rate(data, 'Sex')
    print(survival_rates)
    """
    if category not in data.columns:
        raise KeyError(f"Category {category} not found in DataFrame.")

    survival_rates = {}
    categories = data[category].unique()

    try:
        for cat in categories:
            is_category = data[category] == cat
            survived = data.loc[is_category, 'Survived']
            rate = survived.sum() / survived.count()
            survival_rates[cat] = round(float(rate), 2)
    except ZeroDivisionError:
        raise ZeroDivisionError("Division by zero occurred while calculating survival rates.")

    return survival_rates
    