#!/usr/bin/env python3
"""Figure 2: Cosine Baseline vs Improvement -- 散点图+拟合趋势线"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': False,
    'figure.dpi': 300,
})

# 数据
data = {
    'NFCorpus':  (0.2195, 49.0),
    'SCIDOCS':   (0.1110, 93.5),
    'ArguAna':   (0.3047, 45.0),
    'FiQA':      (0.1683, 136.2),
    'SciFact':   (0.4483, 7.7),
    'Quora':     (0.6370, 6.0),
}

names = list(data.keys())
x = np.array([data[n][0] for n in names])
y = np.array([data[n][1] for n in names])

fig, ax = plt.subplots(figsize=(3.5, 3.0))

# 拟合趋势线（对数拟合更自然）
# 使用 log 拟合: y = a * ln(x) + b
coeffs = np.polyfit(np.log(x), y, 1)
x_fit = np.linspace(0.08, 0.70, 200)
y_fit = coeffs[0] * np.log(x_fit) + coeffs[1]

# 先画趋势线（底层）
ax.plot(x_fit, y_fit, '--', color='#999999', linewidth=1.0, alpha=0.7,
        label='Log fit', zorder=1)

# 散点
colors_pts = ['#4C72B0', '#55A868', '#C44E52', '#DD8452', '#8172B2', '#CCB974']
for i, name in enumerate(names):
    ax.scatter(x[i], y[i], c=colors_pts[i], s=60, zorder=3,
               edgecolors='#333333', linewidths=0.6)

# 标注数据集名（手动微调位置避免重叠）
offsets = {
    'NFCorpus':  (8, 8),
    'SCIDOCS':   (8, -5),
    'ArguAna':   (8, 5),
    'FiQA':      (-15, 10),
    'SciFact':   (8, 5),
    'Quora':     (-12, 8),
}
for i, name in enumerate(names):
    ox, oy = offsets[name]
    ax.annotate(name, (x[i], y[i]),
                xytext=(ox, oy), textcoords='offset points',
                fontsize=7.5, color='#333333',
                arrowprops=dict(arrowstyle='-', color='#aaaaaa',
                                lw=0.5) if abs(ox) > 10 else None)

ax.set_xlabel('Cosine Baseline NDCG@10', fontsize=10)
ax.set_ylabel('Best Improvement (%)', fontsize=10)

# R^2 标注
ss_res = np.sum((y - (coeffs[0] * np.log(x) + coeffs[1]))**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res / ss_tot
ax.text(0.97, 0.95, f'$R^2 = {r2:.3f}$',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=8, color='#555555',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='#cccccc', alpha=0.8))

ax.set_xlim(0.05, 0.72)
ax.set_ylim(-5, 155)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=8)

plt.tight_layout()
fig.savefig('/home/amd/HEZIMENG/legal-assistant/word/arxiv/figures/fig2_scatter.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print('Figure 2 saved.')
