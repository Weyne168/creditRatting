import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dat_file = '../data/explains.ch.npz'
pt = np.load(dat_file, allow_pickle=True)['dict'][()]
pt = pt['ex'][0].transpose(1, 0)

print(pt.shape)
# exit(11)
# pt=pt[11:-11,11:-11]
pt2 = np.concatenate([pt[:, :5] * 255, pt[:, -5:] * 255], axis=1)
pt2 = pt2.astype(np.int)
# pt2 =pt2.transpose(1, 0)
print(pt2.shape)
print(pt2)
# exit(12)
# max_d = np.max(pt)
# min_d = np.min(pt)
# pt = (pt - min_d) / (max_d - min_d)

fig, ax1 = plt.subplots(figsize=(5, 10), ncols=1)

# fig = plt.figure(figsize=(40, 40))

# cmap用cubehelix map颜色
cmap = sns.cubehelix_palette(n_colors=255, start=1.5, rot=3, gamma=0.8, as_cmap=True)
# pt = np.random.rand(112, 112)
# pt = np.random.rand(4, 3)
h = sns.heatmap(pt2, linewidths=0.5, ax=ax1, vmax=255, vmin=0, cmap=cmap, cbar=False)
im = ax1.imshow(pt2, cmap)

# ax1.set_title('Explanations\n', fontsize=120)
# ax1.set_xlabel('Time')
ax1.set_xticklabels(['f1,f2,f7', 'f24', 'f9,f6', 'f1,f1', 'f3,f6'], rotation=90)
# ax1.set_ylabel('Time', fontsize=50)
# ax1.set_ylabel('Time')
ax1.set_yticklabels(['t1', 't2', 't3', 't4', 't5'], rotation=0)
# plt.tick_params(labelsize=80)
ax1.xaxis.set_ticks_position('top')
cb = fig.colorbar(im, fraction=0.046, pad=0.04, orientation='horizontal')
cb.ax.tick_params(labelsize=50)

'''

# cmap用matplotlib colormap
sns.heatmap(pt, linewidths=0.05, ax=ax2, vmax=900, vmin=0, cmap='rainbow')
# rainbow为 matplotlib 的colormap名称
ax2.set_title('matplotlib colormap')
ax2.set_xlabel('region')
ax2.set_ylabel('kind')
'''
fig.savefig('./test3.eps')
