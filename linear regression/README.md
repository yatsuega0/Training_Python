# 線形回帰(linear regression)

予測値が線形になるという仮定を置く機械学習の手法

<img src="https://latex.codecogs.com/svg.image?f(X)=\theta_0&plus;\theta_1X_1&plus;\cdot\cdot\cdot&plus;\theta_nX_n&plus;\epsilon" />

$f(X)$ : 真の関数(目的変数)

$\theta_j$ : j番目のパラメータ(変数)

$X_j$ : j番目の特徴量の値

$\epsilon$ : 誤差

最適な線形(の式)を求める方法に最小二乗法(least squares)がある。
最小二乗法とは残差の二乗和を最初にするアルゴリズムであり、実際の値と予測値のズレの二乗和をを損失と捉えこの損失が最小になるように最適なパラメータθを求めることになる。


<img src="https://latex.codecogs.com/svg.image?\sum_{i=1}^{m}e_i^2=\sum_{i=1}^{m}\left\{y_i-(\theta_0&plus;\theta_1x_i)\right\}^2">



$\theta_0+\theta_1x_i$ : 線形モデルの予測値

$x$ : 特徴量の値

$m$ : データ数

$i$ : i番目のデータ

$y_i$ : i番目のデータの実際の値(真の値)

$\hat{y}$ : i番目のデータの予測値
$e_i$ : i番目の残差(実際の値と予測値の差: $y_i-\hat{y_i}$)


予測値と実際の値のズレを求める関数のことを損失関数といい機械学習の世界ではこの損失関数を最小にする最適なパラメータを見つける。








