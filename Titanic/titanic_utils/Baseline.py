import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore")

# Examples
# import Baseline as bl
# X, y, test_X = bl.create_data(df_train, df_test)


def create_data(train, test):
    """
    データを作成し、学習用とテスト用の特徴量及びラベルを抽出。

    Parameters:
    -----------
    train : DataFrame
        学習用データセット
    test : DataFrame
        テスト用データセット

    Returns:
    --------
    X : ndarray
        学習用特徴量
    y : Series
        学習用ラベル
    test_X : ndarray
        テスト用特徴量

    Examples:
    ---------
    X, y, test_X = create_data(train_data, test_data)
    """
    y_col = 'Survived'
    X = train[['Pclass', 'Parch']].values
    y = train[y_col]
    test_X = test[['Pclass', 'Parch']].values
    
    return X, y, test_X


def split_train_test(X, y, test_size=0.2, random_state=0):
    """
    学習データとテストデータに分割。

    Parameters:
    -----------
    X : ndarray
        特徴量の配列
    y : Series
        ラベルの配列
    test_size : float
        テストデータの比率。
    random_state : int
        乱数のシード値

    Returns:
    --------
    X_train : ndarray
        学習用の特徴量
    X_test : ndarray
        テスト用の特徴量
    y_train : Series
        学習用のラベル
    y_test : Series
        テスト用のラベル

    Examples:
    ---------
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
    
def model_svc(X_train, y_train, test_X, random_state=0):
    """
    SVCモデルを訓練し、予測結果を返す。

    Parameters:
    -----------
    X_train : ndarray
        学習用の特徴量
    y_train : Series
        学習用のラベル
    test_X : ndarray
        テスト用の特徴量
    random_state : int
        乱数のシード値

    Returns:
    --------
    y_pred : ndarray
        テストデータの予測ラベル

    Examples:
    ---------
    y_pred = model_svc(X_train, y_train, test_X)
    """
    svc = SVC(random_state=random_state)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(test_X)    

    return y_pred

def model_lgbm(X_train, y_train, test_X):
    """
    LightGBMモデルを訓練し、予測結果を返す。

    Parameters:
    -----------
    X_train : ndarray
        学習用特徴量
    y_train : Series
        学習用ラベル
    test_X : ndarray
        テスト用特徴量
    params : dict
        YMLファイルから読み込んだLightGBMのパラメータ

    Returns:
    --------
    y_pred : ndarray
        テストデータの予測ラベル

    Examples:
    ---------
    y_pred = model_lgbm(X_train, y_train, test_X)
    """
    # LightGBMのパラメータを定義
    with open('../settings/lgbm_parameters.yml', 'r') as file:
        params = yaml.safe_load(file)
        
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(test_X)
    

    return y_pred

def result_to_csv(y_pred, output_name):
    """
    予測結果を「gender_submission.csv」に代入してCSVファイルに出力。

    Parameters:
    -----------
    y_pred : ndarray
        予測ラベル
    output_name : str
        出力するファイル名

    Returns:
    --------
    None

    Examples:
    ---------
    result_to_csv(test_data, y_pred)
    """
    output = pd.read_csv('../data/gender_submission.csv')
    output['Survived'] = y_pred.astype("int32")
    output.to_csv(f"../output/{output_name}_result.csv", index=False)

    return "gender_submission.csv updated with predictions"
