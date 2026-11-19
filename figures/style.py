import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'dejavuserif', 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7})
OFF, OFF2, OURS, INK, MUTED = ('#c4462d', '#e3a898', '#2a78d6', '#1d1d1f', '#6b6b70')
LS = {'fp32': '-', 'fp16': '--', 'bf16': ':'}
