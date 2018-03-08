from __future__ import print_function

import numpy    as np

from    itertools   import combinations
from    collections import defaultdict
import  time, pickle, os.path, sys
import  matplotlib.pyplot as plt

from    scipy.cluster           import hierarchy
from    scipy.spatial.distance  import squareform


fn = os.path.expanduser('~/Data/Neuro/Arno/training1_ASD_binarized.csv')
a  = np.loadtxt(fn, delimiter=',', dtype='i1')

def pairwise_similarity(a):
    similarity = np.zeros(shape=(a.shape[0], a.shape[0]), dtype='i4')
    for i in range(a.shape[0]):
        for j in range(i, a.shape[0]):
            difference = np.sum(np.logical_xor(a[i], a[j]))
            similarity[i,j] = difference
            similarity[j,i] = difference
    return similarity

transpose = True
prefix = 'questions'
if transpose:
    a = a.T    # we are interested in column simplices
    prefix = 'people'

print(f'{prefix} are points')

similarity = pairwise_similarity(a)

#for clustering_type in ['single','complete','average','weighted','centroid','ward']:
clustering_type = 'ward'
Z  = hierarchy.linkage(squareform(similarity), clustering_type)

print(f'Using {clustering_type} clustering')
# Visualize the dendrogram
dn = hierarchy.dendrogram(Z)
#plt.savefig(f'{prefix}-dendrogram-{clustering_type}.pdf')
#plt.clf()
plt.show()

cutoff = 800        # choose the threshold from the dendrogram
clusters = hierarchy.fcluster(Z, cutoff, criterion = 'distance')
#print(clusters)     # point-to-cluster

labels = np.unique(clusters)
for l in labels:
    print(f'Cluster {l}: {np.where(clusters == l)[0]}')        # cluster-to-points

