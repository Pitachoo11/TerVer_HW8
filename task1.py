import numpy as np
import pandas as pd

zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110]
ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]

# Находим ковариацию с помощью элементарных действий
mean_zp = sum(zp) / len(zp)
mean_ks = sum(ks) / len(ks)
covariance = sum((zp[i] - mean_zp) * (ks[i] - mean_ks) for i in range(len(zp))) / (len(zp) - 1)

# Находим ковариацию с помощью функции cov из numpy
covariance_np = np.cov(zp, ks, ddof=1)[0, 1]

# Проверяем, что значения ковариации, найденные двумя способами, совпадают
print(f'Ковариация, найденная элементарными действиями: {covariance:.2f}')
print(f'Ковариация, найденная функцией cov из numpy: {covariance_np:.2f}')
assert np.isclose(covariance, covariance_np)

# Находим коэффициент корреляции Пирсона
std_zp = np.std(zp, ddof=1)
std_ks = np.std(ks, ddof=1)
print(f"Среднеквадратичное отклонение заработной платы: {std_zp}")
print(f"Среднеквадратичное отклонение кредитного скоринга: {std_ks}")

pearson = covariance / (std_zp * std_ks)
print(f'Коэффициент корреляции Пирсона (ковариация и среднеквадратичные отклонения): {pearson}')

#Находим коэффициент корреляции Пирсона с помощью pandas
data = pd.DataFrame({'zp': zp, 'ks': ks})

corr_matrix = data.corr(method='pearson')
corr_coef = corr_matrix.iloc[0, 1]

print(f'Коэффициент корреляции Пирсона (pandas): {corr_coef}')

#Находим ковариации и коэффициент корреляции Пирсона с помощью numpy
covariance = np.cov(zp, ks, ddof=1)[0, 1]
print(f"Ковариация: {covariance}")

corr_coef = np.corrcoef(zp, ks)[0, 1]
print(f"Коэффициент корреляции Пирсона: {corr_coef}")