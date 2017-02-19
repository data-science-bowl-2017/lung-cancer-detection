
# coding: utf-8

# In[92]:

import numpy as np
import pandas as pd
import matplotlib
import pylab
import glob
import dicom
import cv2
import os
import matplotlib.pyplot as plt
from skimage import data, io, filters

from IPython.display import HTML
import matplotlib.animation as animation

get_ipython().magic(u'matplotlib nbagg')
get_ipython().magic(u'matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")


# In[80]:

labels = pd.read_csv('../../stage1_labels.csv')
labels.head()
labels[labels['id'] == '0d941a3ad6c889ac451caf89c46cb92a'] # No Cancer


# In[101]:

paths = glob.glob(os.path.join('*'))
labels[[s in paths for s in labels['id']]]


# In[102]:

os.chdir("/Users/ianjohnson/Desktop/data-science-bowl/sample_images/")
print(os.getcwd())
os.chdir('0acbebb8d463b4b9ca88cf38431aac69')
print(os.listdir('.')[:5])
files = os.listdir('.')

imgs = []
for i in files:
    try:
        ds = dicom.read_file(i)
        imgs.append(ds)
    except Exception as e:
        pass


# In[103]:

#sorting based on InstanceNumber stolen from r4m0n's script: 
imgs.sort(key = lambda x: int(x.InstanceNumber))
full_img = np.stack([s.pixel_array for s in imgs])


# In[104]:

bool = full_img[:,2,:] > -2000
full_img[:,2,:][bool]
full_img.shape


# In[110]:

fig = plt.figure()
im = plt.imshow(full_img[100,:,:], cmap=pylab.cm.bone)


# In[112]:

imgs_train = np.load("../../ionox0/trainImages.npy").astype(np.float32)
imgs_train.shape


# In[114]:

#from https://www.kaggle.com/z0mbie/data-science-bowl-2017/chest-cavity-animation-with-pacemaker
#not working..hmm

fig = plt.figure()
im = plt.imshow(imgs_train[0,0,:,:], cmap=pylab.cm.bone)

# Function to update figure
def updatefig(j):
    # set the data in the axesimage object
    im.set_array(imgs_train[j,0,:,:])
    # return the artists set
    return im,

# Kick off the animation
ani = animation.FuncAnimation(fig, updatefig, frames=range(len(full_img)), 
                              interval=50, blit=True)
# ani.save('Chest_Cavity.gif', writer='imagemagick')
plt.show()


# In[ ]:

for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.imshow(full_img[4 * i, :, :], cmap=plt.cm.bone)    
    plt.xticks([])
    plt.yticks([])


# In[ ]:

for i in range(36):
    plt.subplot(6, 6, i + 1)
    img = cv2.resize(full_img[:, 20 + 12 * i, :], (256, 256))
    plt.imshow(img, cmap=plt.cm.bone)
    plt.xticks([])
    plt.yticks([])


# In[ ]:

for i in range(36):
    plt.subplot(6,6,i+1)
    img = cv2.resize(full_img[:, :, 20 + 12 * i], (256, 256))
    plt.imshow(img, cmap=plt.cm.bone)
    plt.xticks([])
    plt.yticks([])


# In[ ]:



