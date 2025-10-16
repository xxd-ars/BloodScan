import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# =====================
# Data
# =====================
metrics = ["DSR", "IoU", "mAP", "AP75", "AP50", "F1", "Precision", "Recall"]

yolo11_white = [37.32, 79.3, 37.02, 33.30, 52.13, 51.02, 61.76, 48.15]
yolo11_blue = [21.67, 88.3, 77.98, 81.08, 97.93, 86.54, 82.72, 94.38]
dual_concat = [68.15, 90.0, 84.18, 85.32, 96.25, 90.25, 88.20, 92.78]
dual_weighted = [91.57, 89.3, 83.12, 79.36, 98.02, 96.06, 95.49, 96.64]
dual_crossattn = [98.80, 90.3, 84.93, 84.68, 99.50, 99.40, 99.68, 99.12]

# =====================
# Custom transform for y-axis: compress 0-80, expand 80-100
# =====================
def forward_transform(y):
    """Transform: compress 0-80 into 0-40, expand 80-100 into 40-100"""
    return np.where(y <= 80, y * 0.5, 40 + (y - 80) * 3)

def inverse_transform(y_trans):
    """Inverse transform"""
    return np.where(y_trans <= 40, y_trans * 2, 80 + (y_trans - 40) / 3)

# =====================
# Plot style setup
# =====================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['font.size'] = 13
plt.rcParams['mathtext.fontset'] = 'stix'

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(metrics))

# Transform data
yolo11_white_t = forward_transform(np.array(yolo11_white))
yolo11_blue_t = forward_transform(np.array(yolo11_blue))
dual_concat_t = forward_transform(np.array(dual_concat))
dual_weighted_t = forward_transform(np.array(dual_weighted))
dual_crossattn_t = forward_transform(np.array(dual_crossattn))

# Plot transformed data
ax.plot(x, yolo11_white_t, '-^', color='#e377c2', linewidth=2.5, markersize=8,
        label='Yolo11-White', markeredgecolor='black', markeredgewidth=0.8)
ax.plot(x, yolo11_blue_t, '-s', color='#1f77b4', linewidth=2.5, markersize=8,
        label='Yolo11-Blue', markeredgecolor='black', markeredgewidth=0.8)
ax.plot(x, dual_concat_t, '-D', color='#2ca02c', linewidth=2.5, markersize=8,
        label='Dual Yolo Channel Concat', markeredgecolor='black', markeredgewidth=0.8)
ax.plot(x, dual_weighted_t, '-o', color='#ff7f0e', linewidth=2.5, markersize=8,
        label='Dual Yolo Adaptive Weighted', markeredgecolor='black', markeredgewidth=0.8)
ax.plot(x, dual_crossattn_t, '-p', color='#9467bd', linewidth=2.5, markersize=8,
        label='Dual Yolo Cross Attention', markeredgecolor='black', markeredgewidth=0.8)

# =====================
# Custom y-axis labels
# =====================
y_ticks_orig = [0, 20, 40, 60, 80, 85, 90, 95, 100]
y_ticks_trans = forward_transform(np.array(y_ticks_orig))
ax.set_yticks(y_ticks_trans)
ax.set_yticklabels([f'{int(y)}' for y in y_ticks_orig], fontsize=12)

# Add horizontal line at 80% to show transition
ax.axhline(y=forward_transform(80), color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(len(metrics)-0.5, forward_transform(80), '80%', fontsize=10, color='gray',
        ha='right', va='bottom')

# =====================
# Aesthetics
# =====================
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=13, fontweight='bold')
ax.set_ylabel('Average Metric Value (%)', fontsize=14, fontweight='bold')
ax.set_ylim(forward_transform(0), forward_transform(101))
ax.grid(True, linestyle='--', alpha=0.4, linewidth=1)
ax.legend(frameon=True, loc='lower right', fontsize=11, framealpha=0.3,
          edgecolor='black', fancybox=False)

# Add note about scale
ax.text(0.02, 0.98, 'Note: Y-axis scale expanded above 80%',
        transform=ax.transAxes, fontsize=10, style='italic',
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# =====================
# Save high-DPI PNG for publication
# =====================
plt.savefig('paper/paper_tex/img/metrics.png', dpi=600, bbox_inches='tight', facecolor='white')
print("✓ Saved: metrics.png (600 DPI)")

plt.close()
