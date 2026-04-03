#!/usr/bin/env python3
"""Figure 1: Pipeline Architecture -- 四阶段管线架构图
   使用全宽7寸图，确保所有文字完整显示"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': False,
    'figure.dpi': 300,
})

fig, ax = plt.subplots(figsize=(4.5, 5.2))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(0.5, 14.5)
ax.axis('off')
ax.set_aspect('auto')

# 颜色方案
colors = {
    'query':  '#4C72B0',
    'stage1': '#55A868',
    'stage2': '#C44E52',
    'stage3': '#8172B2',
    'stage4': '#CCB974',
    'text':   '#333333',
}

def draw_box(ax, cx, cy, w, h, line1, line2, color, fs1=9, fs2=7):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=color, edgecolor='#333333',
                         linewidth=1.0, alpha=0.88, zorder=2)
    ax.add_patch(box)
    if line2:
        ax.text(cx, cy + 0.17, line1, ha='center', va='center',
                fontsize=fs1, fontweight='bold', color='white', zorder=3)
        ax.text(cx, cy - 0.22, line2, ha='center', va='center',
                fontsize=fs2, color='#f0f0f0', style='italic', zorder=3)
    else:
        ax.text(cx, cy, line1, ha='center', va='center',
                fontsize=fs1, fontweight='bold', color='white', zorder=3)

def draw_arrow(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color='#555555',
                                lw=1.3), zorder=1)

# 布局参数
cx = 5.0
bw = 5.8
bh = 1.1
ys = [13.5, 11.3, 9.1, 6.9, 4.7]  # 自上而下5个位置

# --- Query ---
draw_box(ax, cx, ys[0], bw, 0.85,
         'Query Input', '6 tokens, 4096d each', colors['query'],
         fs1=9, fs2=7)

draw_arrow(ax, cx, ys[0] - 0.43, ys[1] + 0.55)
ax.text(cx + 0.15, (ys[0] + ys[1]) / 2,
        'N documents', ha='left', va='center', fontsize=7, color='#777777')

# --- Stage 1 ---
draw_box(ax, cx, ys[1], bw, bh,
         'Stage 1: Centroid Filtering',
         'cosine similarity  |  23 ms', colors['stage1'],
         fs1=8.5, fs2=6.5)

draw_arrow(ax, cx, ys[1] - 0.55, ys[2] + 0.55)
ax.text(cx + 0.15, (ys[1] + ys[2]) / 2,
        u'N \u2192 top-200', ha='left', va='center', fontsize=7, color='#777777')

# --- Stage 2 (长标题，缩小字号) ---
draw_box(ax, cx, ys[2], bw, bh,
         'Stage 2: PQ-Chamfer Rerank',
         u'64 subspaces \u00d7 64d  |  23 ms', colors['stage2'],
         fs1=8, fs2=6.5)

draw_arrow(ax, cx, ys[2] - 0.55, ys[3] + 0.55)
ax.text(cx + 0.15, (ys[2] + ys[3]) / 2,
        u'top-200 \u2192 top-55', ha='left', va='center', fontsize=7, color='#777777')

# --- Stage 3 (长标题，缩小字号) ---
draw_box(ax, cx, ys[3], bw, bh,
         'Stage 3: Graph Smoothing',
         'KNN(K=3) + Laplacian diffusion  |  26 ms', colors['stage3'],
         fs1=8, fs2=6.5)

draw_arrow(ax, cx, ys[3] - 0.55, ys[4] + 0.55)
ax.text(cx + 0.15, (ys[3] + ys[4]) / 2,
        'score propagation', ha='left', va='center', fontsize=7, color='#777777')

# --- Stage 4 ---
draw_box(ax, cx, ys[4], bw, bh,
         'Stage 4: Score Fusion',
         u'0.7 \u00d7 token + 0.3 \u00d7 graph \u2192 top-10', colors['stage4'],
         fs1=8.5, fs2=6.5)

# 总延迟
ax.text(cx, 3.1, 'Total latency: ~72 ms per query',
        ha='center', va='center', fontsize=8,
        fontweight='bold', color=colors['text'],
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                  edgecolor='#bbbbbb', linewidth=0.8))

fig.savefig('/home/amd/HEZIMENG/legal-assistant/word/arxiv/figures/fig1_pipeline.pdf',
            bbox_inches='tight', pad_inches=0.05, dpi=300)
plt.close()
print('Figure 1 saved.')
