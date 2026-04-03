#!/usr/bin/env python3
"""Figure 4: PQ-Chamfer Subspace Decomposition 示意图"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'text.usetex': False,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# 颜色 -- 用渐变色系表示不同子空间
cmap = plt.cm.Set3
sub_colors = [cmap(i / 8) for i in range(8)]

# --- 顶部：4096d 向量 ---
y_top = 7.8
vec_w = 10.0
vec_h = 0.6
vec_x0 = 2.0

# 大框
rect = FancyBboxPatch((vec_x0, y_top - vec_h/2), vec_w, vec_h,
                       boxstyle="round,pad=0.05",
                       facecolor='#E8E8E8', edgecolor='#333333',
                       linewidth=1.2)
ax.add_patch(rect)
ax.text(vec_x0 + vec_w/2, y_top, '4096-dimensional embedding vector',
        ha='center', va='center', fontsize=9, fontweight='bold',
        color='#333333')

# "split" 标注
ax.annotate('', xy=(7.0, 6.7), xytext=(7.0, 7.5),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))
ax.text(7.55, 7.1, 'split into 64\nsubspaces', ha='left', va='center',
        fontsize=7.5, color='#666666', style='italic')

# --- 中间行：子空间方块 ---
y_sub = 6.2
num_shown = 6  # 显示6个子空间 + 省略号
sub_w = 1.4
sub_h = 0.7
sub_gap = 0.25
total_w = num_shown * sub_w + (num_shown - 1) * sub_gap + 1.0  # +1 for ellipsis
x_start = (14 - total_w) / 2

sub_labels = ['sub$_1$\n64d', 'sub$_2$\n64d', 'sub$_3$\n64d',
              'sub$_4$\n64d', '...', 'sub$_{64}$\n64d']

sub_x_centers = []
for i, label in enumerate(sub_labels):
    if label == '...':
        x = x_start + i * (sub_w + sub_gap) + sub_w / 2
        ax.text(x, y_sub, '...', ha='center', va='center',
                fontsize=14, fontweight='bold', color='#888888')
        sub_x_centers.append(x)
        continue
    x = x_start + i * (sub_w + sub_gap)
    ci = i if i < 4 else (i + 58)  # 颜色索引
    rect = FancyBboxPatch((x, y_sub - sub_h/2), sub_w, sub_h,
                           boxstyle="round,pad=0.05",
                           facecolor=sub_colors[ci % len(sub_colors)],
                           edgecolor='#555555', linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + sub_w/2, y_sub, label, ha='center', va='center',
            fontsize=7.5, color='#333333')
    sub_x_centers.append(x + sub_w / 2)

# --- 子空间 cosine 距离 ---
y_cos = 5.0
for i, xc in enumerate(sub_x_centers):
    if sub_labels[i] == '...':
        ax.text(xc, y_cos, '...', ha='center', va='center',
                fontsize=14, fontweight='bold', color='#888888')
        continue
    # 箭头
    ax.annotate('', xy=(xc, y_cos + 0.25), xytext=(xc, y_sub - sub_h/2 - 0.05),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))
    # cos 标签
    ax.text(xc, y_cos, 'cos', ha='center', va='center',
            fontsize=7, color='#555555',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#cccccc', linewidth=0.5))

# --- delta 值行 ---
y_delta = 4.1
delta_labels = [r'$\delta_1(a,b)$', r'$\delta_2(a,b)$', r'$\delta_3(a,b)$',
                r'$\delta_4(a,b)$', '...', r'$\delta_{64}(a,b)$']
for i, xc in enumerate(sub_x_centers):
    if delta_labels[i] == '...':
        ax.text(xc, y_delta, '...', ha='center', va='center',
                fontsize=14, fontweight='bold', color='#888888')
        continue
    ax.annotate('', xy=(xc, y_delta + 0.25), xytext=(xc, y_cos - 0.25),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))
    ax.text(xc, y_delta, delta_labels[i], ha='center', va='center',
            fontsize=8, color='#333333')

# --- 汇聚箭头到 average ---
y_avg = 2.6
avg_x = 7.0

# 从每个 delta 画汇聚线
for i, xc in enumerate(sub_x_centers):
    if delta_labels[i] == '...':
        continue
    ax.annotate('', xy=(avg_x, y_avg + 0.4),
                xytext=(xc, y_delta - 0.25),
                arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=0.6,
                                connectionstyle='arc3,rad=0'))

# Average 框
avg_w = 3.5
avg_h = 0.65
rect = FancyBboxPatch((avg_x - avg_w/2, y_avg - avg_h/2), avg_w, avg_h,
                       boxstyle="round,pad=0.1",
                       facecolor='#4C72B0', edgecolor='#333333',
                       linewidth=1.2, alpha=0.85)
ax.add_patch(rect)
ax.text(avg_x, y_avg, 'average over 64 subspaces',
        ha='center', va='center', fontsize=8.5,
        fontweight='bold', color='white')

# --- 最终结果 ---
y_result = 1.4
ax.annotate('', xy=(avg_x, y_result + 0.35), xytext=(avg_x, y_avg - avg_h/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

result_w = 3.0
result_h = 0.55
rect = FancyBboxPatch((avg_x - result_w/2, y_result - result_h/2),
                       result_w, result_h,
                       boxstyle="round,pad=0.1",
                       facecolor='#C44E52', edgecolor='#333333',
                       linewidth=1.2, alpha=0.85)
ax.add_patch(rect)
ax.text(avg_x, y_result, r'$d_{\mathrm{PQ}}(a, b)$',
        ha='center', va='center', fontsize=10,
        fontweight='bold', color='white')

# --- 右侧注释：VT-Aligned 变体 ---
note_x = 12.0
note_y = 3.5
ax.text(note_x, note_y + 0.8, 'VT-Aligned:', ha='center', va='center',
        fontsize=7.5, fontweight='bold', color='#8172B2')
ax.text(note_x, note_y + 0.3, r'min-then-average', ha='center', va='center',
        fontsize=7, color='#8172B2')
ax.text(note_x, note_y - 0.15, r'($\min$ per subspace,', ha='center', va='center',
        fontsize=6.5, color='#999999')
ax.text(note_x, note_y - 0.5, r'then average)', ha='center', va='center',
        fontsize=6.5, color='#999999')
ax.text(note_x, note_y - 0.95, '+2.5% gain', ha='center', va='center',
        fontsize=7, fontweight='bold', color='#55A868')

# 虚线框
rect_note = FancyBboxPatch((note_x - 1.5, note_y - 1.3), 3.0, 2.5,
                            boxstyle="round,pad=0.1",
                            facecolor='none', edgecolor='#8172B2',
                            linewidth=0.8, linestyle='--')
ax.add_patch(rect_note)

plt.tight_layout()
fig.savefig('/home/amd/HEZIMENG/legal-assistant/word/arxiv/figures/fig4_pq_decomp.pdf',
            bbox_inches='tight', dpi=300)
plt.close()
print('Figure 4 saved.')
