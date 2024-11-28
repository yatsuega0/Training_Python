import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import japanize_matplotlib


df = pd.read_csv("Fielder_Data2023.csv")

threshold_plate_appearances = 400

# 今回は打席数400以上の選手を抽出
df = df[df['打席数'] >= threshold_plate_appearances]

# OPS 出塁率+長打率
df["OPS"] = df["出塁率"] + df["長打率"]

# カラムを選手名と本塁打とOPSに絞る
df = df[["選手名", "本塁打", "OPS"]]

# 散布図
sns.scatterplot(x=df['本塁打'], y=df['OPS'])
plt.show()

# 最小二乗法
x = df['本塁打'].values
y = df['OPS'].values
theta_0 = 2
theta_1 = 2
# 各データにおける残差（正解値と予測値のズレ）の二乗和
print((y-(theta_0 + theta_1 * x))**2)
# 損失関数を求める
np.mean((y-(theta_0 + theta_1 * x))**2)


# 損失関数
def loss_func(theta_0, theta_1, x, y):
    return np.mean((y-(theta_0 + theta_1 * x))**2)


# 最急降下法
# θ0とθ1の初期値を設定
theta_0_init = -5
theta_1_init = 5

# イテレーションの数
epoch = 10**5

# 学習率
alpha = 0.0001


# θ0のパラメータ更新
def update_theta0(theta_0, theta_1, x, y, alpha=0.001):
    return theta_0 - alpha * 2 * np.mean((theta_0 + theta_1 * x) - y)


# θ1のパラメータ更新
def update_theta1(theta_0, theta_1, x, y, alpha=0.001):
    return theta_1 - alpha * 2 * np.mean(((theta_0 + theta_1 * x) - y) * x)


# パラメータをイテレーションで更新
theta_0_hist = []
theta_1_hist = []
# 初期化
theta_0_hist.append(theta_0_init)
theta_1_hist.append(theta_1_init)

for _ in range(epoch):
    update_theta_0 = update_theta0(theta_0_hist[-1], theta_1_hist[-1], x=x, y=y, alpha=alpha)
    update_theta_1 = update_theta1(theta_0_hist[-1], theta_1_hist[-1], x=x, y=y, alpha=alpha)
    theta_0_hist.append(update_theta_0)
    theta_1_hist.append(update_theta_1)

# 最適なパラメータを使用した回帰モデルの描画
sns.scatterplot(x=df['本塁打'], y=df['OPS'])
x_values = np.arange(55)
y_values = theta_0_hist[-1] + theta_1_hist[-1]*x_values
plt.plot(x_values, y_values, '-', color='r')
plt.show()

if __name__ == "__main__":
    loss_func(2, 2, x, y)
    update_theta0(theta_0_init, theta_1_init, x, y)
    update_theta1(theta_0_init, theta_1_init, x, y)
    print("θ0", theta_0_hist[-1])
    print("θ1", theta_1_hist[-1])
    # 最急降下法の推移
    ops_cost = [loss_func(*param, x=x, y=y) for param in zip(theta_0_hist, theta_1_hist)]
    print(ops_cost[-1])

    # 予測
    # 山田　哲人 本塁打:14 OPS:0.721
    Home_Run = 14
    OPS = theta_0_hist[-1] + theta_1_hist[-1]*Home_Run
    print("予測", OPS)
    # 0.746090318083526
