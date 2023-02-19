import numpy as np
from scipy.stats import t

# Значения выборки
x = np.array([131, 125, 115, 122, 131, 115, 107, 99, 125, 111])

# Среднее и стандартное отклонение по выборке
x_mean = np.mean(x)
x_std = np.std(x, ddof=1)

# Определяем значение степеней свободы и альфа
n = len(x)
df = n - 1
alpha = 1 - 0.95

# Определяем значение t-критерия для надежности 95%
t_critical = t.ppf(1 - alpha/2, df)

# Вычисление доверительного интервала
lower = x_mean - t_critical * x_std / np.sqrt(n)
upper = x_mean + t_critical * x_std / np.sqrt(n)

print("Доверительный интервал, содержащий истинное значение математического ожидания с надежностью 0.95: ({:.2f}, {:.2f})".format(lower, upper))
