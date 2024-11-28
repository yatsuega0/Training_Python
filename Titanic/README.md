
# Titanic Project

## 概要
Titanicの乗客データを用いて生存予測モデルを開発。
データの前処理からモデリング、結果の可視化までの一連のプロセスが含まれる。

## ディレクトリ構造

```
Titanic/
│
├── titanic_utils/
│   ├── Baseline.py           # ベースラインモデルの作成スクリプト
│   ├── Data_check.py         # データ読み込みと確認スクリプト
│   ├── Modeling_rf.py        # ランダムフォレストでの学習と推論スクリプト
│   ├── Processing.py         # データ前処理と特徴量生成スクリプト
│   └── Visualization.py      # データ可視化スクリプト
│
├── notebooks/
│   ├── Baseline.ipynb        # ベースラインモデルの実行ノートブック
│   ├── Visualization.ipynb   # データ可視化実行ノートブック
│   └── Modeling.ipynb        # モデル構築と予測実行ノートブック
│
├── settings/
│   ├── lgbm_parameters.yml   # LightGBMのパラメータ設定ファイル
│   ├── ListofTitles.xlsx     # 名称グループ作成用リスト
│   └── parameters.yml        # モデルパラメータ設定ファイル
│
├── output/                   # モデルの出力と予測結果を格納
│
└── data/                     # 入力データを格納
```

## ファイル説明
- **Baseline.py**: 初期モデルとしてのベースラインを設定し推論を行う。
- **Data_check.py**: 生データの読み込みと基本的なデータ確認を実施する。
- **Modeling_rf.py**: ランダムフォレストを使用したモデル学習と推論のプロセスを含む。
- **Processing.py**: 必要なデータクレンジング、前処理、および特徴量エンジニアリングを行う。
- **Visualization.py**: 分析結果を視覚的に表示し、洞察を得るためのグラフやチャートを作成。

## 使用方法
各Pythonスクリプトは、対応するノートブックから呼び出されることを想定している。詳細な使用方法については、各スクリプトの説明およびドキュメントを参照。
