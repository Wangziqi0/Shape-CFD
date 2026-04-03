#!/usr/bin/env python3
"""Figure 3: top_n vs NDCG@10 曲线 -- NFCorpus + FiQA 双数据集"""

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
top_n_vals = [55, 100, 200]

nfcorpus = [0.2900, 0.2936, 0.2946]
fiqa     = [0.2426, 0.2557, 0.2661]

fig, ax1 = plt.subplots(figsize=(3.5, 2.8))

color_nf = '#4C72B0'
color_fq = '#C44E52'

# NFCorpus (左Y轴)
ln1 = ax1.plot(top_n_vals, nfcorpus, 'o-', color=color_nf,
               linewidth=1.5, markersize=5, label='NFCorpus')
ax1.set_xlabel('Candidate Set Size (top-$n$)', fontsize=10)
ax1.set_ylabel('NDCG@10 (NFCorpus)', fontsize=9, color=color_nf)
ax1.tick_params(axis='y', labelcolor=color_nf, labelsize=8)
ax1.tick_params(axis='x', labelsize=8)

# FiQA (右Y轴)
ax2 = ax1.twinx()
ln2 = ax2.plot(top_n_vals, fiqa, 's--', color=color_fq,
               linewidth=1.5, markersize=5, label='FiQA')
ax2.set_ylabel('NDCG@10 (FiQA)', fontsize=9, color=color_fq)
ax2.tick_params(axis='y', labelcolor=color_fq, labelsize=8)

# 合并图例
lns = ln1 + ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='lower right', fontsize=8,
           frameon=True, framealpha=0.9, edgecolor='#cccccc')

# 标注数据点值（手动控制偏移避免重叠）
nf_offsets = [(0, 8), (0, 8), (-20, 8)]    # top-200 偏左
fq_offsets = [(0, -12), (0, -12), (20, -12)]  # top-200 偏右
for i, v in enumerate(nfcorpus):
    ax1.annotate(f'{v:.4f}', (top_n_vals[i], v),
                 xytext=nf_offsets[i], textcoords='offset points',
                 fontsize=7, ha='center', color=color_nf)

for i, v in enumerate(fiqa):
    ax2.annotate(f'{v:.4f}', (top_n_vals[i], v),
                 xytext=fq_offsets[i], textcoords='offset points',
                 fontsize=7, ha='center', color=color_fq)

# X轴刻度
ax1.set_xticks(top_n_vals)
ax1.set_xticklabels(['55', '100', '200'])

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

# 设置合理Y轴范围
ax1.set_ylim(0.285, 0.300)
ax2.set_ylim(0.230, 0.275)

plt.tight_layout()
fig.savefig('/home/amd/HEZIMENG/legal-assistant/word/arxiv/figures/fig3_topn.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print('Figure 3 saved.')
