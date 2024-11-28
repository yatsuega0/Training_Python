import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def fill_missing_values(df, column, value='Unknown'):
    """
    データフレームの指定されたカラムの欠損値を、指定された値で補完する関数。

    Parameters
    ----------
    df : pd.DataFrame
        欠損値を補完する対象のデータフレーム
    column : str
        欠損値を補完するカラム名
    value : any, optional
        欠損値を補完するために使用する値。デフォルトは 'Unknown'

    Returns
    -------
    df : pd.DataFrame
        欠損値が補完されたデータフレーム

    Examples
    --------
    >>> df = fill_missing_values(df, 'Name')

    >>> df = fill_missing_values(df, 'Age', df['Age'].median())
    """
    df[column] = df[column].fillna(value)
    return df


def add_family_column(df):
    """
    データフレームに 'Family' 列を追加する関数です。
    'Parch'（両親/子供の数）と 'SibSp'（配偶者/兄弟の数）の列の値を合計し、
    新しい列 'Family' をデータフレームに追加。

    Parameters
    ----------
    df : pd.DataFrame
        'Parch' と 'SibSp' 列を含む pandas データフレーム

    Returns
    -------
    df : pd.DataFrame
        'Family' 列が追加されたデータフレーム

    Examples
    --------
    >>> df = add_family_column(df)
    """
    df['Family'] = df['Parch'] + df['SibSp']
    return df

def label_ticket_groups(df, ticket_column='Ticket'):
    """
    チケットの文字数に基づいてデータフレーム内の生存率グループをラベル付けする関数。

    Parameters
    ----------
    df : pd.DataFrame
        チケット情報を含むデータフレーム
    ticket_column : str, optional
        チケット情報が含まれるカラム名。デフォルトは 'Ticket'

    Returns
    -------
    df : pd.DataFrame
        'TicketGroup' カラムが追加されたデータフレーム

    Examples
    --------
    >>> df = label_ticket_groups(df)
    """
    Ticket_Count = dict(df[ticket_column].value_counts())
    
    def ticket_label(s):
        if (s >= 2) & (s <= 4):
            return 2
        elif ((s > 4) & (s <= 8)) | (s == 1):
            return 1
        elif s > 8:
            return 0

    df['TicketGroup'] = df[ticket_column].apply(lambda x: Ticket_Count[x])
    df['TicketGroup'] = df['TicketGroup'].apply(ticket_label)

    return df

def assign_honorific_groups(df, title_ref_path='../settings/ListofTitles.xlsx'):
    """
    DataFrame の 'Name' 列から敬称を抽出し、事前に定義された敬称リストを使用し
    対応する '敬称グループ' を 'Honorifics' 列として追加する関数。

    Parameters
    ----------
    df : pd.DataFrame
        敬称を抽出する対象のデータフレーム
    title_ref_path : str, optional
        敬称リストが保存されているExcelファイルのパス

    Returns
    -------
    df : pd.DataFrame
        'Honorifics' 列が追加されたデータフレーム。
        欠損値は 'Unknown'で補完

    Examples
    --------
    >>> df = assign_honorific_groups(df)
    """
    # 敬称リストを読み込む
    titles_df = pd.read_excel(title_ref_path)
    title_dict = titles_df.set_index('Name')['敬称グループ'].to_dict()

    # 敬称を抽出して敬称グループをマッピング
    df['Honorifics'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    df['Honorifics'] = df['Honorifics'].map(title_dict)
    df['Honorifics'] = df['Honorifics'].fillna('Unknown')
    
    return df


def predict_and_fill_age(df, columns=['Age', 'Pclass', 'Sex', 'Honorifics']):
    """
    DataFrame 内の年齢の欠損値について、
    ランダムフォレスト回帰モデルを使用して予測し、代入する関数。

    Parameters
    ----------
    df : pd.DataFrame
        年齢の欠損値を補完する対象のデータフレーム
    columns : list, optional
        年齢予測に使用するカラムのリスト

    Returns
    -------
    df : pd.DataFrame
        年齢の欠損値が補完されたデータフレーム。

    Examples
    --------
    >>> df = predict_and_fill_age(df)
    """

    # 利用するカラムでダミー変数を作成
    age = df[columns]
    age_dummies = pd.get_dummies(age)

    # 年齢がわかるデータと欠損しているデータに分ける
    known_age = age_dummies[age_dummies['Age'].notnull()]
    null_age = age_dummies[age_dummies['Age'].isnull()]

    age_X = known_age.drop('Age', axis=1).values
    age_y = known_age['Age'].values

    # ランダムフォレスト回帰モデルで年齢予測
    rf = RandomForestRegressor()
    rf.fit(age_X, age_y)
    predicted_ages = rf.predict(null_age.drop('Age', axis=1).values)

    df.loc[df['Age'].isnull(), 'Age'] = predicted_ages

    return df

import pandas as pd

def assign_family_labels(df, sibsp_col='SibSp', parch_col='Parch'):
    """
    家族のサイズを計算し、指定された規則に基づいてラベルを割り当てる関数。

    Parameters
    ----------
    df : pd.DataFrame
        処理対象のデータフレーム
    sibsp_col : str, optional
        兄弟姉妹の数が含まれるカラム名
        デフォルトは 'SibSp'。
    parch_col : str, optional
        両親と子供の数が含まれるカラム名
        デフォルトは 'Parch'

    Returns
    -------
    df : pd.DataFrame
        'FamilySize' と 'FamilyLabel' 列が追加されたデータフレーム。

    Examples
    --------
    >>> df = assign_family_labels(df)
    """

    # 家族サイズの計算
    df['FamilySize'] = df[sibsp_col] + df[parch_col] + 1

    # 家族サイズに基づいてラベルを割り当てる
    def family_label(s):
        if 2 <= s <= 4:
            return 2
        elif 1 == s or 5 <= s <= 7:
            return 1
        elif s > 7:
            return 0

    df['FamilyLabel'] = df['FamilySize'].apply(family_label)

    return df



if __name__ == '__main__':
    # df_train = ps.fill_missing_values(df_train, 'Age', value=df_train['Age'].median())
    df_train = ps.fill_missing_values(df_train, 'Fare', value=df_train['Fare'].median())
    df_train = ps.fill_missing_values(df_train, 'Embarked')

    # df_train = ps.add_family_column(df_train)

    df_train = ps.label_ticket_groups(df_train)

    df_train = ps.assign_honorific_groups(df_train)

    df_train = ps.predict_and_fill_age(df_train)

    df_train = ps.assign_family_labels(df_train)


