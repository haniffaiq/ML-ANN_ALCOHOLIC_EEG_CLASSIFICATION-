import scipy.io
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from PIL import Image

# Mendapatkan data dari file .mat
data = scipy.io.loadmat('EEG64Kanal.mat')

# Mendapatkan list array dari data
list_array = data['SimpanGambar']

data_dict = {}

# Mengubah list array menjadi bentuk baris
row_array = np.vstack(list_array)

# Membuat dataframe dari array
df = pd.DataFrame(row_array, columns=['Matrix'])

# Menampilkan dataframe
print(df)
# Menambahkan kolom 'Label' dengan nilai default 'Non_Alcoholic'
df['Label'] = 'Non_Alcoholic'

# Mengatur label 'Alcoholic' pada data ke-1 hingga ke-600
df.loc[:599, 'Label'] = 'Alcoholic'

# Mengubah dataframe menjadi gambar dan menyimpannya ke dalam folder "Data_Normal"
for index, row in df.iterrows():
    # Membuat gambar dari data dalam baris
    img_array = np.array(row['Matrix'], dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Menyimpan gambar ke dalam folder "Data_Normal" dengan label sebagai nama file
    label = row['Label']
    img.save('Data_Normal/{}_{}.png'.format(label, index))
