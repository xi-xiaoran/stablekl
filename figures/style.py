"""Shared figure style: closed axes, inward ticks on all sides, Times-like fonts, no point markers on dense curves."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
    'axes.linewidth': 0.6, 'axes.labelsize': 8, 'axes.titlesize': 8.3, 'legend.fontsize': 6.6, 'legend.frameon': False,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.minor.width': 0.4, 'ytick.minor.width': 0.4, 'xtick.minor.size': 1.4, 'ytick.minor.size': 1.4,
})
# official = red, ours = blue, official formula in the next higher precision = green
OFF, OURS, PROM, INK, MUTED = ('#E64B35', '#2a78d6', '#00A087', '#1d1d1f', '#6b6b70')
OFF2 = PROM
DASH = (0, (4, 2))
LS = {'fp32': '-', 'fp16': DASH, 'bf16': (0, (1, 1.3))}
import os
if os.environ.get('KLFIG_GRID') == '1':   # optional light background grid
    plt.rcParams.update({'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.5, 'axes.axisbelow': True})
