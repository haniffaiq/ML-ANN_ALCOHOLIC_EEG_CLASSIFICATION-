#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scipy.io
import pandas as pd
import numpy as np
from google.colab import drive
import cv2
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt

drive.mount('/content/drive',force_remount=True)

# In[ ]:


dir = '/content/drive/MyDrive/Tugas Akhir/Donny/'
items = os.listdir(dir)
items.sort()

# In[ ]:


from random import sample
test = sample(items,1)
test

# In[ ]:


from google.colab.patches import cv2_imshow
from PIL import Image

img = cv2.imread('/content/drive/MyDrive/Tugas Akhir/Datasethisto/'+test[0])
print(test[0])


plt.imshow(img)
plt.axis('off')


# In[ ]:


import numpy as np

def ciriGLDM(x):
    # menghitung ciri dari GLDM
    # h(g|tetha) = probabilitas dif
    # Gradien kontras = sigma (g(h(q|theta)))

    z = np.histogram(np.uint8(x), bins=256, range=(0, 256))[0]
    hg = z / np.sum(z) # Normalized histogram

    # Initialize feature vectors
    gradkon = np.zeros(256) # Gradient contrast
    gradmean = np.zeros(256) # Gradient mean
    idm = np.zeros(256) # Inverse different moment

    # Calculate features for each gray level
    for i in range(256):
        gradkon[i] = (((i-1)**2) * hg[i]) # Gradient contrast for each gray level
        gradmean[i] = (i-1) * hg[i] # Gradient mean for each gray level
        idm[i] = hg[i] / ((i-1)**2 + 1) # Inverse different moment for each gray level

    gradkont = np.sum(gradkon) # Sum of gradient contrasts (ASM)
    gradsm = np.sum(hg**2) # Sum of squared probabilities (second moment)
    gradent = np.sum(hg * np.log10(hg + np.finfo(float).eps)) # Entropy
    gradmeant = np.sum(gradmean) # Sum of gradient means
    idmt = np.sum(idm) # Sum of inverse different moments

    return gradkont, gradsm, gradent, gradmeant, idmt

# In[ ]:


from tqdm.notebook import tqdm_notebook
new_path = "/content/drive/MyDrive/Tugas Akhir/Datasethisto/"
items = os.listdir(new_path)
items.sort()
distance = 5
for dis in tqdm_notebook(range(distance),"distance"):
  # Initialize variables
  FiturEEG = np.zeros((1250, 25))
  N = 0
  d = dis+1
  print(d)

  for k in tqdm_notebook(items,"Image"):
      image = cv2.imread(new_path+k)

      s = np.shape(image)
      inImg = image.astype(float)

      # matrices
      pro1 = np.zeros(s)  # diff arah 0 derajat
      pro2 = np.zeros(s)  # diff arah 45 derajat
      pro3 = np.zeros(s)  # diff arah 90 derajat
      pro4 = np.zeros(s)  # diff arah 135 derajat

      for i in range(s[0]-d): # dikurangin d agar index tidak melebihi dimensi
          for j in range(s[1]-d): # dikurangin d agar index tidak melebihi dimensi

              if (j + d) <= s[1]:
                  pro1[i, j] = abs(inImg[i, j] - inImg[i, j + d])
              if (i - d) > 0 and (j + d) <= s[1]:
                  pro2[i, j] = abs(inImg[i, j] - inImg[i - d, j + d])
              if (i + d) <= s[0]:
                  pro3[i, j] = abs(inImg[i, j] - inImg[i + d, j])
              if (i - d) > 0 and (j - d) > 0:
                  pro4[i, j] = abs(inImg[i, j] - inImg[i - d, j - d])

      # ekstraksi ciri
      gradkont0, gradsm0, gradent0, gradmeant0, idmt0 = ciriGLDM(pro1)
      ciri0 = [gradkont0, gradsm0, gradent0, gradmeant0, idmt0]

      gradkont45, gradsm45, gradent45, gradmeant45, idmt45 = ciriGLDM(pro2)
      ciri45 = [gradkont45, gradsm45, gradent45, gradmeant45, idmt45]

      gradkont90, gradsm90, gradent90, gradmeant90, idmt90 = ciriGLDM(pro3)
      ciri90 = [gradkont90, gradsm90, gradent90, gradmeant90, idmt90]

      gradkont135, gradsm135, gradent135, gradmeant135, idmt135 = ciriGLDM(pro4)
      ciri135 = [gradkont135, gradsm135, gradent135, gradmeant135, idmt135]

      kontrata = (gradkont0 + gradkont45 + gradkont90 + gradkont135) / 4
      asmrata = (gradsm0 + gradsm45 + gradsm90 + gradsm135) / 4
      entrata = (gradent0 + gradent45 + gradent90 + gradent135) / 4
      meanrata = (gradmeant0 + gradmeant45 + gradmeant90 + gradmeant135) / 4
      idmrata = (idmt0 + idmt45 + idmt90 + idmt135) / 4
      cirirata = [kontrata, asmrata, entrata, meanrata, idmrata]

      ciritotal = np.concatenate((ciri0, ciri45, ciri90, ciri135, cirirata))

      FiturEEG[N, :] = ciritotal

      N = N + 1
  columns_feature = 'gradkont0, gradsm0, gradent0, gradmeant0, idmt0,gradkont45, gradsm45, gradent45, gradmeant45, idmt45,gradkont90, gradsm90, gradent90, gradmeant90, idmt90,gradkont135, gradsm135, gradent135, gradmeant135, idmt135,kontrata, asmrata, entrata, meanrata, idmrata'
  columns_feature = columns_feature.split(',')
  columns_feature = [col+"_distance_"+str(d) for col in columns_feature]
  # columns_feature = columns_feature.split(" ")
  features = pd.DataFrame(FiturEEG,columns=columns_feature)
  features.to_csv("/content/drive/MyDrive/Tugas Akhir/GLDM_Histo/"+str(d)+".csv")

