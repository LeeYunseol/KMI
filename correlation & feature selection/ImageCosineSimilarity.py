from img2vec_keras import Img2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
img2vec = Img2Vec()
target = img2vec.get_vec('graph_2018/SCFI.jpg')
feature1 = img2vec.get_vec('graph_2018/Average Earnings.jpg')
feature2 = img2vec.get_vec('graph_2018/PCI- South East Asia.jpg')
print(feature1)
X = np.stack([feature1, feature2, target])
Y = X
similarity_matrix = cosine_similarity(X, Y)

print(similarity_matrix)

#%%
#%%
def cos_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    l2_norm = (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))
    similarity = dot_product / l2_norm     
    similarity = round(similarity, 2)
    return similarity

print(cos_similarity(feature2, target))
print(cos_similarity(feature1, target))
print(cos_similarity(target, target))

feature3 = img2vec.get_vec('graph_2018/Newbuilding Prices(3500,4000 TEU) with -8 week lag.jpg')
print(cos_similarity(feature3, target))
#%%
# 2018년 기준 그래프 이미지 분석
columns = ['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom,Continent',
       'PCI- Mediterranean,Black Sea', 'PCI- East Asia',
       'PCI- South East Asia', 'PCI- China P.R.', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650,1850 TEU)',
       'Newbuilding Prices(13000,14000 TEU)',
       'Newbuilding Prices(3500,4000 TEU)',
       'Newbuilding Prices(13000,13500 TEU)', '5 Year Finance based on Libor',
       'Exchange Rates South Korea', 'Exchange Rates Euro',
       'Exchange Rates China']

new_columns = ['SCFI', 'PCI-Comprehensive', 'PCI- East Coast North America',
       'PCI- West Coast North America', 'PCI- United Kingdom,Continent',
       'PCI- Mediterranean,Black Sea', 'PCI- East Asia',
       'PCI- South East Asia', 'PCI- China P.R.', 'Average Earnings',
       'Bunker Prices', 'Total Container ships Number',
       'Total Container ships TEU', 'Newbuilding Prices(1650,1850 TEU)',
       'Newbuilding Prices(13000,14000 TEU)',
       'Newbuilding Prices(3500,4000 TEU)',
       'Newbuilding Prices(13000,13500 TEU)', '5 Year Finance based on Libor',
       'Exchange Rates South Korea', 'Exchange Rates Euro',
       'Exchange Rates China']


lag_list = [-8, -4, +4, +8]
for column in tqdm(columns) :
    for lag in lag_list :
        if (lag > 0) :
            new_column = column + ' with ' + "+" + str(lag) +' week lag'
        else :
            new_column = column + ' with ' + str(lag) +' week lag'
        new_columns.append(new_column)  

target = img2vec.get_vec('graph_2018/SCFI.jpg')

for column in new_columns :
    feature_path = 'graph_2018/' + column + '.jpg'
    feature = img2vec.get_vec(feature_path)
    print(column, '과 SCFI의 img2vec를 활용한 코사인 유사도는 ', cos_similarity(feature, target))

#%%
# 코랩 
from img2vec_keras import Img2Vec

from IPython.display import Image

import glob

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import cv2

image_vectors = {}
for column in new_columns : 
    feature_path = 'graph_2018/' + column + '.jpg'
    feature = img2vec.get_vec(feature_path)
    image_vectors[feature_path] = feature
    
X = np.stack(list(image_vectors.values()))

pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(X)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
print(np.shape(pca_result_50))

    
tsne = TSNE(n_components=2, verbose=1, n_iter=3000)
tsne_result = tsne.fit_transform(pca_result_50)

tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

plt.scatter(tsne_result_scaled[:,0], tsne_result_scaled[:,1])
#%%
images = []
for column in new_columns : 
  feature_path = 'graph_2018/' + column + '.jpg'
  image = cv2.imread(feature_path, 3)
  b,g,r = cv2.split(image)           # get b, g, r
  image = cv2.merge([r,g,b])         # switch it to r, g, b
  image = cv2.resize(image, (200, 200))
  images.append(image)        

fig, ax = plt.subplots(figsize=(20,15))
artists = []

for xy, i in zip(tsne_result_scaled, images):
  x0, y0 = xy
  img = OffsetImage(i, zoom=.7)
  ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
  artists.append(ax.add_artist(ab))
ax.update_datalim(tsne_result_scaled)
ax.autoscale(enable=True, axis='both', tight=True)
plt.show()