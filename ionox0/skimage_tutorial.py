
# coding: utf-8

# In[16]:

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction

# Convert to float: Important for subtraction latiner which won't work with uint8
image = img_as_float(data.coins())
image = gaussian_filter(image, 1)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')


# In[17]:

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5), sharex=True, sharey=True)

ax1.imshow(image)
ax1.set_title('original image')
ax1.axis('off')
ax1.set_adjustable('box-forced')

ax2.imshow(dilated, vmin=image.min(), vmax=image.max())
ax2.set_title('dilated')
ax2.axis('off')
ax2.set_adjustable('box-forced')

ax3.imshow(image - dilated)
ax3.set_title('image - dilated')
ax3.axis('off')
ax3.set_adjustable('box-forced')

fig.tight_layout()


# In[19]:

plt.show()`


# In[20]:

h = 0.4
seed = image - h
dilated = reconstruction(seed, mask, method='dilation')
hdome = image - dilated


# In[21]:

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 2.5))

yslice = 197

ax1.plot(mask[yslice], '0.5', label='mask')
ax1.plot(seed[yslice], 'k', label='seed')
ax1.plot(dilated[yslice], 'r', label='dilated')
ax1.set_ylim(-0.2, 2)
ax1.set_title('image slice')
ax1.set_xticks([])
ax1.legend()

ax2.imshow(dilated, vmin=image.min(), vmax=image.max())
ax2.axhline(yslice, color='r', alpha=0.4)
ax2.set_title('dilated')
ax2.axis('off')

ax3.imshow(hdome)
ax3.axhline(yslice, color='r', alpha=0.4)
ax3.set_title('image - dilated')
ax3.axis('off')

fig.tight_layout()
plt.show()


# In[53]:

x = np.linspace(0, 4 * np.pi)
x_plot = plt.plot(x, label='x_plot')

y_mask = np.cos(x)
y_mask_plot = plt.plot(y_mask, label='y_mask')

y_seed = y_mask.min() * np.ones_like(x)
y_seed_plot = plt.plot(y_seed, label='y_seed')

y_seed[0] = 0.5
y_seed[-1] = 0
y_rec = reconstruction(y_seed, y_mask)
y_rec_plot = plt.plot(y_rec, label='y_rec')

plt.legend([x_plot],['x_plot']) #, y_mask_plot, y_seed_plot, y_rec_plot])
plt.show()


# In[ ]:




# In[ ]:



