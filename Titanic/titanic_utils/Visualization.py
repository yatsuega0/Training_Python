import matplotlib.pyplot as plt


DICT_SURVIVED = {0: '0: Dead', 1: '1: Survived'}

def arrange_stack_bar(ax):
    """
    X軸のラベルを回転し、グリッドを点線で描画。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        設定を適用するmatplotlibのAxesオブジェクト

    Returns
    -------
    None
    """
    ax.set_xticklabels(labels=ax.get_xticklabels(), rotation=30, horizontalalignment="center")
    ax.grid(axis='y', linestyle='dotted')

def output_bars(df, column, index={}):
    """
    指定されたデータフレームからカテゴリ別に生死の割合を表示するパイチャートとスタックバーチャートを作成し、表示。

    Parameters
    ----------
    df : pandas.DataFrame
        データを含むpandas DataFrame。'Survived'と指定されたカラムを持つ必要がある
    column : str
        グループ化の基準となるカテゴリカルデータのカラム名
    index : dict, optional
        プロットする際にカテゴリの値をリネームするための辞書
        デフォルトは空の辞書

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)    

    # Key-Valueラベルなしの場合
    if len(index) == 0:
        df_vc = df.groupby([column])["Survived"].value_counts(
            sort=False).unstack().rename(columns=DICT_SURVIVED)
        df[column].value_counts().plot.pie(ax=axes[0, 0], autopct="%1.1f%%")
        df.groupby([column])["Survived"].value_counts(
            sort=False, normalize=True).unstack().rename(columns=DICT_SURVIVED).plot.bar(ax=axes[1, 1], stacked=True)
    
    # Key-Valueラベルありの場合
    else:
        df_vc = df.groupby([column])["Survived"].value_counts(
            sort=False).unstack().rename(index=index, columns=DICT_SURVIVED)
        df[column].value_counts().rename(index).plot.pie(ax=axes[0, 0], autopct="%1.1f%%")
        df.groupby([column])["Survived"].value_counts(
            sort=False, normalize=True).unstack().rename(index=index, columns=DICT_SURVIVED).plot.bar(ax=axes[1, 1], stacked=True)   

    df_vc.plot.bar(ax=axes[1, 0])

    for rect in axes[1, 0].patches:
        height = rect.get_height()
        axes[1, 0].annotate('{:.0f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    df_vc.plot.bar(ax=axes[0, 1], stacked=True)

    arrange_stack_bar(axes[0, 1])
    arrange_stack_bar(axes[1, 0])
    arrange_stack_bar(axes[1, 1])

    # データラベル追加
    [axes[0, 1].text(i, item.sum(), item.sum(), horizontalalignment='center') 
     for i, (_, item) in enumerate(df_vc.iterrows())]

    plt.show()


def output_box_hist(df, column, bins=20, query=None):
    """
    指定されたカラムに対するボックスプロットとヒストグラムを描画。
    オプショナルで、特定のクエリに基づいたデータのフィルタリングが可能。

    Parameters
    ----------
    df : pandas.DataFrame
        描画するデータフレーム
    column : str
        描画するデータのカラム名。
    bins : int, optional
        ヒストグラムのビンの数でデフォルトは20
    query : str, optional
        データをフィルタリングするためのクエリ文字列
        デフォルトはNoneですべてのデータが対象

    Returns
    -------
    None

    Notes
    -----
    欠損値はこの関数では除去されている
    そのため、関数の入力として渡されるデータには欠損値が含まれていないことを前提としている
    """
    if query is None:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    else:
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
        df.query(query)[column].hist(ax=axes[2, 0], bins=bins)
        df.query(query).groupby('Survived')[column].plot.hist(
        ax=axes[2, 1], bins=bins, alpha=0.5, legend=True, grid=True)
        axes[2, 1].legend(labels=[DICT_SURVIVED[int(float((text.get_text())))] for text in axes[2, 1].get_legend().get_texts()])

    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    df.boxplot(ax=axes[0, 0], column=[column])
    df.boxplot(ax=axes[0, 1], column=[column], by='Survived')
    axes[0, 1].set_xticklabels([DICT_SURVIVED[int(float(xticklabel.get_text()))] for xticklabel in axes[0, 1].get_xticklabels()])
    df[column].hist(ax=axes[1, 0], bins=bins)
    df.groupby('Survived')[column].plot.hist(ax=axes[1, 1], bins=bins, alpha=0.5, grid=True, legend=True)
    axes[1, 1].legend(labels=[DICT_SURVIVED[int(float((text.get_text())))] for text in axes[1, 1].get_legend().get_texts()])

    plt.show()

if __name__ == '__main__' :
	# Pclassの確認
	DICT_PCLASS = {1: '1: 1st(Upper)', 2: '2: 2nd(Middle)', 3: '3: 3rd(Lower)'}
	output_bars(df_train, 'Pclass', DICT_PCLASS)
	# Sexの確認
	output_bars(df_train, 'Sex')
	# Embarkedの確認
	DICT_EMBARK = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
	output_bars(df_train, 'Embarked', DICT_EMBARK)
	# SibSpの確認
	output_bars(df_train, 'SibSp')
	# Parchの確認
	output_bars(df_train, 'Parch')

	# Age
	output_box_hist(df_train, 'Age')
	# Fare
	output_box_hist(df_train, 'Fare', 20, 'Fare < 200')

	# correlation matrix
	df_train.loc[:, ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr().style.background_gradient(axis=None)


