import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 论文频次数据
years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
frequencies = np.array([50, 60, 70, 90, 120, 150, 140, 160, 180, 200])

# 计算均值和标准差
mu, sigma = np.mean(frequencies), np.std(frequencies)

# 生成正态分布概率密度函数
x = np.linspace(min(frequencies), max(frequencies), 1000)
pdf = norm.pdf(x, mu, sigma)

# 绘制频次数据的正态分布图
plt.plot(x, pdf, label='Probability Density Function (PDF)')
plt.hist(frequencies, bins=10, density=True, alpha=0.6, color='g', label='Histogram of Frequencies')
plt.title('Probability Density Function and Frequency Histogram')
plt.xlabel('Number of Papers')
plt.ylabel('Density')
plt.legend()
plt.show()

# 计算区间[2019, 2021]内的概率密度（知识密度）
interval_prob = norm.cdf(2021, mu, sigma) - norm.cdf(2019, mu, sigma)
print(f"Knowledge Density (Probability in [2019, 2021]): {interval_prob}")
