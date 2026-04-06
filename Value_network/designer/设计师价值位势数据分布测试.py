import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("D:\\Documents\\Desktop\\一品威客数据采集4月\\test523\\designer_nodes.csv", encoding='utf-8-sig')
plt.hist(df['NicheWidth'], bins=50, log=True)  # 使用对数刻度查看分布
plt.xlabel('NicheWidth')
plt.ylabel('Frequency (Log Scale)')
plt.title('Distribution of NicheWidth')
plt.show()
print(df['ValuePotential'].describe())  # 查看统计信息