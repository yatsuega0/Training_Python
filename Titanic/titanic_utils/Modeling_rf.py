import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')



def load_parameters(file_path):
    """パラメータ設定ファイルを読み込む
    Args:
        file_path (str): パラメータファイルのパス
    
    Returns:
        dict: 設定されたパラメータ
    """
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params


def train_model(X, y, params):
    """モデルを訓練し、最適なパラメータを見つける
    Args:
        X (pd.DataFrame): 訓練データ
        y (pd.Series): 目標変数
        params (dict): ランダムフォレストの設定
    
    Returns:
        GridSearchCV: 訓練されたグリッドサーチオブジェクト
    """
    pipe = Pipeline([('classify', RandomForestClassifier(random_state=params['random_forest_config']['random_state'],
                                                         max_features=params['random_forest_config']['max_features']))])
    param_test = {
        'classify__n_estimators': params['random_forest_config']['n_estimators'],
        'classify__max_depth': params['random_forest_config']['max_depth']
    }
    gsearch = GridSearchCV(estimator=pipe, param_grid=param_test, scoring='accuracy', cv=10)
    gsearch.fit(X, y)
    return gsearch


def predict(model, test_data):
    """テストデータに対して予測を行う
    Args:
        model (GridSearchCV): 訓練されたモデル
        test_data (pd.DataFrame): テストデータ
    
    Returns:
        np.array: 予測結果
    """
    predictions = model.predict(test_data)
    return predictions

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

# 例: 使用例
if __name__ == "__main__":
    # パラメータの読み込み
    params = load_parameters("settings/parameters.yml")
    
    
    # モデル訓練
    trained_model = train_model(X, y, params)
    
    # モデル評価
    best_params = trained_model.best_params_
    best_score = trained_model.best_score_
    print(f"Best Parameters: {best_params}")
    print(f"Best Score: {best_score}")
    
    # テストデータの予測
    predictions = predict(trained_model, test_data)
    print(predictions)
