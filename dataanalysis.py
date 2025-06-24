import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import DBSCAN

# ===== モデル関数の定義 =====
def exponential_model(x, a, b, c):

    return a * np.exp(-b * x) + c

# ===== クラスタリングと回帰を実施する関数 =====
def plot_centerX_vs_theta_with_evaluation_with_clustering(df, x_range=None, y_range=None, theta_range=None, total_pixels_range=None):
    # 必要なカラムのチェック
    required_columns = ['Theta', 'CenterX - ImageCenterX', 'CenterY - ImageCenterY', 'TotalPixels']
    if not all(col in df.columns for col in required_columns):
        print("必要な列がデータフレームに含まれていません。")
        return

    # 各変数の抽出
    X = df[['CenterX - ImageCenterX']].values      # x軸のデータ
    Y = df['Theta'].values                         # y軸のデータ（Theta）
    Y_center = df['CenterY - ImageCenterY'].values  # 中心位置（参考用）
    total_pixels = df['TotalPixels'].values         # ピクセル数（フィルタリング用）

    # ----- 範囲フィルタリング -----
    mask_x = (X[:, 0] >= x_range[0]) & (X[:, 0] <= x_range[1]) if x_range else np.ones_like(X[:, 0], dtype=bool)
    mask_y = (Y_center >= y_range[0]) & (Y_center <= y_range[1]) if y_range else np.ones_like(Y_center, dtype=bool)
    mask_theta = (Y >= theta_range[0]) & (Y <= theta_range[1]) if theta_range else np.ones_like(Y, dtype=bool)
    mask_pixels = (total_pixels >= total_pixels_range[0]) & (total_pixels <= total_pixels_range[1]) if total_pixels_range else np.ones_like(total_pixels, dtype=bool)

    mask = mask_x & mask_y & mask_theta & mask_pixels
    X_filtered = X[mask]
    Y_filtered = Y[mask]

    # ----- スケーリングとクラスタリング (DBSCAN) -----
    # xとyのスケールが異なる場合に備え、yにスケールファクターを掛ける
    scale_factor = (X_filtered.max() - X_filtered.min()) / (Y_filtered.max() - Y_filtered.min())
    data_scaled = np.column_stack((X_filtered, Y_filtered * scale_factor))
    clustering = DBSCAN(eps=121.95, min_samples=1).fit(data_scaled)
    labels = clustering.labels_

    # クラスタリングにより外れ値を除去（ラベル -1 は外れ値）
    mask_dbscan = labels != -1
    X_cleaned = X_filtered[mask_dbscan]
    Y_cleaned = Y_filtered[mask_dbscan]

    # ----- 2次多項式回帰 -----
    degree = 2  # ここを変更

    poly = PolynomialFeatures(degree=degree)
    X_poly_cleaned = poly.fit_transform(X_cleaned)

    poly_model = LinearRegression()
    poly_model.fit(X_poly_cleaned, Y_cleaned)

    Y_pred_poly = poly_model.predict(X_poly_cleaned)
    residuals_poly = Y_cleaned - Y_pred_poly
    r2_poly = r2_score(Y_cleaned, Y_pred_poly)
    mse_poly = mean_squared_error(Y_cleaned, Y_pred_poly)

    print("---- 2次多項式回帰 ----")
    print(f"定数項: {poly_model.intercept_:.3f}")
    for i in range(1, degree + 1):
        print(f"{i}次項の係数: {poly_model.coef_[i]:.10f}")
    print(f"R²: {r2_poly:.3f}")
    print(f"平均二乗誤差 (MSE): {mse_poly:.3f}")
    print("")

    # プロット（2次多項式回帰）
    plt.figure(figsize=(9,9))
    plt.xlim(x_range)
    plt.ylim(theta_range)
    plt.scatter(X_cleaned, Y_cleaned, color='blue', s=10, alpha=0.8, label="Cleaned Data")
    centerX_range = np.linspace(X_cleaned.min(), X_cleaned.max(), 100).reshape(-1, 1)
    centerX_poly_range = poly.transform(centerX_range)
    theta_pred_poly = poly_model.predict(centerX_poly_range)
    plt.plot(centerX_range, theta_pred_poly, color='orange', linewidth=2, label="2D Fit")
    plt.xlabel('CenterX - ImageCenterX')
    plt.ylabel('Theta (degrees)')
    plt.title('CenterX vs Theta')
    plt.legend()
    plt.grid()
    plt.show()

    # プロット（多項式回帰）
    plt.figure(figsize=(9,9))
    plt.xlim(x_range)
    plt.ylim(theta_range)
    plt.scatter(X_cleaned, Y_cleaned, color='blue', s=10, alpha=0.8, label="Cleaned Data")
    centerX_range = np.linspace(X_cleaned.min(), X_cleaned.max(), 100).reshape(-1, 1)
    centerX_poly_range = poly.transform(centerX_range)
    theta_pred_poly = poly_model.predict(centerX_poly_range)
    plt.plot(centerX_range, theta_pred_poly, color='orange', linewidth=2, label="3D Fit")
    plt.xlabel('CenterX - ImageCenterX')
    plt.ylabel('Theta (degrees)')
    plt.title('CenterX vs Theta')
    plt.legend()
    plt.grid()
    plt.show()

    # 残差プロット（多項式回帰）
    plt.figure(figsize=(7,7))
    plt.scatter(Y_pred_poly, residuals_poly, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Theta (Polynomial)')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid()
    plt.show()

    # ----- 指数関数フィッティング -----
    # 初期パラメータの設定（データに合わせて調整してください）
    initial_guess = [1.0, 0.001, 0.0]

    try:
        popt_exp, pcov_exp = curve_fit(
            exponential_model,
            X_cleaned.ravel(),
            Y_cleaned,
            p0=initial_guess,
            maxfev=10000  # ここに maxfev を追加
        )
    except RuntimeError as e:
        print("指数関数フィッティングに失敗しました:", e)
        return

    a_exp, b_exp, c_exp = popt_exp
    print("---- 指数関数フィッティング ----")
    print(f"a: {a_exp:.3f},  b: {b_exp:.3f},  c: {c_exp:.3f}")

    Y_pred_exp = exponential_model(X_cleaned, *popt_exp)
    r2_exp = r2_score(Y_cleaned, Y_pred_exp)
    mse_exp = mean_squared_error(Y_cleaned, Y_pred_exp)
    std_res_exp = np.std(Y_cleaned - Y_pred_exp)

    print(f"R²: {r2_exp:.3f}")
    print(f"平均二乗誤差 (MSE): {mse_exp:.3f}")
    print(f"残差の標準偏差: {std_res_exp:.3f}")

    # プロット（指数関数フィッティング）
    plt.figure(figsize=(9,9))
    plt.xlim(x_range)
    plt.ylim(theta_range)
    plt.scatter(X_cleaned, Y_cleaned, color='blue', s=10, alpha=0.8, label="Cleaned Data")
    theta_exp_range = exponential_model(centerX_range, *popt_exp)
    plt.plot(centerX_range, theta_exp_range, color='green', linewidth=2, label="Fit")
    plt.xlabel('CenterX - ImageCenterX')
    plt.ylabel('Theta (degrees)')
    plt.title('CenterX - Theta ')
    plt.legend()
    plt.grid()
    plt.show()

    # 残差プロット（指数関数フィッティング）
    plt.figure(figsize=(7,7))
    plt.scatter(Y_pred_exp, Y_cleaned - Y_pred_exp, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel('Predicted Theta (Exponential)')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot (Exponential Fit)')
    plt.grid()
    plt.show()

# ===== エクセルデータの読み込み =====
input_excel = "30degblender.xlsx"
# 複数シートある場合はすべて読み込み、空シートは除外
df_sheets = pd.read_excel(input_excel, sheet_name=None)
df_list = [sheet for sheet in df_sheets.values() if not sheet.empty]
df_combined = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# ===== フィルタリング条件の設定 =====
x_range = (-2500, 2500)
y_range = (-2000, 2000)
theta_range = (1, 90)
total_pixels_range = (1, 5000000)

# ===== 関数の実行 =====
plot_centerX_vs_theta_with_evaluation_with_clustering(
    df_combined,
    x_range=x_range,
    y_range=y_range,
    theta_range=theta_range,
    total_pixels_range=total_pixels_range
)
