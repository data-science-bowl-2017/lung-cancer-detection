
# coding: utf-8

# In[30]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
get_ipython().magic(u'matplotlib inline')

p = sns.color_palette()

os.listdir('./')

seed = 42


# In[6]:

for d in os.listdir('./sample_images'):
    if '.DS_Store' not in d:
        print("Patient '{}' has {} scans".format(d, len(os.listdir('./sample_images/' + d))))
print('----')
print('Total patients {} Total DCM files {}'.format(len(os.listdir('./sample_images')), 
                                                      len(glob.glob('./sample_images/*/*.dcm'))))


# In[7]:

patient_sizes = [len(os.listdir('./sample_images/' + d)) for d in os.listdir('./sample_images') if '.DS_Store' not in d]
plt.hist(patient_sizes, color=p[2])
plt.ylabel('Number of patients')
plt.xlabel('DICOM files')
plt.title('Histogram of DICOM count per patient')


# # Training Set

# In[8]:

df_train = pd.read_csv('./stage1_labels.csv')
df_train.head()


# In[9]:

print('Number of training patients: {}'.format(len(df_train)))
print('Cancer rate: {:.4}%'.format(df_train.cancer.mean()*100))


# # Naive Submission
# Since the evaluation metric used in this competition is LogLoss and not something like AUC, this means that we can often gain an improvement just by aligning the probabilities of our sample submission to that of the training set.
# Before I try making a naive submission, I will calculate what the score of this submission would be on the training set to get a comparison.

# In[10]:

from sklearn.metrics import log_loss
logloss = log_loss(df_train.cancer, np.zeros_like(df_train.cancer) + df_train.cancer.mean())
print('Training logloss is {}'.format(logloss))


# In[11]:

# sample = pd.read_csv('./stage1_sample_submission.csv')
# sample['cancer'] = df_train.cancer.mean()
# sample.to_csv('naive_submission.csv', index=False)


# # Looking at images

# In[12]:

import dicom


# In[13]:

dcm = './0a67f9edb4915467ac16a565955898d3.dcm'
print('Filename: {}'.format(dcm))
dcm = dicom.read_file(dcm)


# In[14]:

print dcm


# In[15]:

img = dcm.pixel_array
print dcm.SliceLocation  
img[img == -2000] = 0
print img

plt.axis('off')
plt.imshow(img)
plt.show()

plt.axis('off')
plt.imshow(-img) # Invert colors with -
plt.show()


# In[16]:

labels = pd.read_csv('stage1_labels.csv')
print labels.head()


# In[17]:

# import csv
# features = open('features.csv', 'wb')
# wr = csv.writer(features)


# In[39]:

a = []
b = []
ids = []
imgs = []
for i in os.listdir('./sample_images/'):
    if '.DS_Store' not in i and '0b20184e0cd497028bdd155d9fb42dc9' not in i:
        print "starting with new patient", i
        label = int(labels[labels['id'] == i]['cancer'])
#         print label
        for j in os.listdir('./sample_images/'+i):
            ids.append(i)
            imgs.append(j)
            b.append(label)
            dcm = dicom.read_file('./sample_images/'+i+'/'+j)
            img = dcm.pixel_array
            img[img == -2000] = 0
            img = np.sum(img, axis=1) ## sum of each row
#             print img.shape
            img = list(img.flatten())
            loca = float(dcm.SliceLocation)
            img.extend([loca])
            a.append(img)

#             break
#         break
df = pd.DataFrame()
df['id'] = ids
df['image'] = imgs
df['label'] = b
df['feat'] = a

X = a
Y = b 


# In[19]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


# In[41]:

# df = pd.read_csv('./features.csv')
# X = np.arrray(df)
print len(df)


# In[89]:

# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=seed)

patients = [i for i in os.listdir('./sample_images/') if '.DS_Store' not in i]
print len(patients)
patients_train = np.random.choice(patients, 15)

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(len(df)):
    if df['id'][i] in patients_train:
        X_train.append(np.array(df['feat'][i]))
        y_train.append(df['label'][i])
    else:
        X_test.append(np.array(df['feat'][i]))
        y_test.append(df['label'][i])


# In[90]:

model = RandomForestClassifier(random_state=seed)
model.fit(X_train, y_train)
predicted = model.predict(X_test)
predicted_prob = model.predict_proba(X_test)
# print 'log loss', log_loss(y_test, predicted)
print 'accuracy score', accuracy_score(y_test, predicted)

print predicted, y_test, X[0]


# In[72]:

# sample = pd.read_csv('./stage1_sample_submission.csv')
# test_ids = sample['id']
# print test_ids
# print np.sum(predicted_prob)
# predicted = model.predict(X_test)
# sample['cancer'] = 
# sample.to_csv('submission2.csv', index=False)


# In[40]:

df.head(10)


# In[ ]:




# In[ ]:



