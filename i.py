import numpy as np
import matplotlib.pyplot as plt

# Data
labels = ['ml-100k', 'ml-1m']
x = np.arange(len(labels))
width = 0.2

svd = [0.9106, 0.8680]
svdpp = [0.9102, 0.8660]
nmf_b = [1.4630, 0.9511]
nmf_u = [0.9573, 0.9056]

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, svd, width, label='SVD')
rects2 = ax.bar(x + width/2, svdpp, width, label='SVD++')
rects3 = ax.bar(x + 3*width/2, nmf_b, width, label='NMF(Biased)')
rects4 = ax.bar(x + 5*width/2, nmf_u, width, label='NMF(Unbiased)')

ax.set_ylabel('RMSE')
ax.set_title('RMSE Comparison for Algorithms')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.savefig('./bar_chart.png')
plt.show()
