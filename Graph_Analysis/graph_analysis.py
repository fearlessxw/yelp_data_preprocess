import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import datetime
import community
import community.community_louvain
from community.community_louvain import modularity
import easygraph as eg
import random
import copy
from sklearn.cluster import AffinityPropagation


pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# load the dataset
print("loading the dataset...")
path = '../yelp_dataset/'
users = pd.read_json(path + 'yelp_academic_dataset_user.json', orient='columns', lines=True)
users = users[["user_id", "friends"]]
total_users = users.shape[0]
print(users.shape)
# print(users.head())

print("getting the whole user graph...")

# add nodes
G = eg.Graph()
G.add_nodes_from(users['user_id'])
print("adding nodes...")

# add edges
edge_list = []


def relationship(row):
    if row['friends'] != 'None':
        edge_list.extend([(row['user_id'], x.strip()) for x in str(row['friends']).split(',')])

users.apply(relationship, axis=1)
print("size of edge_list:" + str(len(edge_list)))
print("adding edges into the graph...")
G.add_edges_from(edge_list)


# CDF of degree

# degree
print("getting the degree...")
degree = list(dict(G.degree()).values())

# only take the degree<100
degree = [x for x in degree if x <= 100]
min_degree = min(degree)
print(min_degree)
print(degree.count(min_degree))
res_freq = stats.relfreq(degree, numbins=100)
cdf_value = np.cumsum(res_freq.frequency)
cdf_value = cdf_value*100
x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
plt.xlabel("Degree")
plt.ylabel("CDF(%)")
plt.ylim(0, 100)
plt.xlim(0, 100)
plt.plot(x, cdf_value)
plt.savefig('CDF_of_degree.png')
plt.show()



# Connected Components

print('getting major connected components ...')
connected_components = list(nx.connected_components(G))
num = []
for i in range(len(connected_components)):
    num.append(len(connected_components[i]))
num.sort(reverse=True)

print('The number of connected components in the raw network:'+str(len(num)))
users_num = G.number_of_nodes()
print("total users: {}".format(users_num))
for i in range(10):
    print("Component {}: {} users".format(i, num[i]))

plt.xlabel("Top 10 connected component")
plt.ylabel("Component size")
plt.yscale("log")
x = ['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th']
plt.bar(x, num[0:10])
plt.savefig('size_of_top10_connected_components.png')
plt.show()





# CDF of clustering coefficient

print("getting the subgraph...")
# sample
sub_users = users.sample(frac=0.3, replace=False, axis=0, random_state=None)
sub_G = G.subgraph(sub_users['user_id'])
print(sub_G.number_of_nodes())
print(sub_G.number_of_edges())

print("getting clustering coefficient...")
# pre_time = datetime.datetime.now()
# get CC of the whole network
CC = nx.clustering(sub_G)
CC = list(dict(CC).values())

min_CC = min(CC)
print("min CC:" + str(min(CC)))
print("node at min CC:" + str(CC.count(min_CC)))

res_freq = stats.relfreq(CC, numbins=100)
cdf_value = np.cumsum(res_freq.frequency)
cdf_value = cdf_value*100
x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
plt.xlabel("Clustering Coefficient")
plt.ylabel("CDF(%)")
plt.xlim(xmin=0)
plt.ylim(0, 100)
plt.plot(x, cdf_value)
plt.savefig('CDF_of_CC.png')
plt.show()

print("average clustering coefficient: {}".format(np.mean(CC)))


# Modularity

# Louvain Algorithm
print("Applying Louvain Algorithm...")
partition = community.community_louvain.best_partition(G)
modularity = modularity(partition, G)
print("modularity:" + str(modularity))


# CDF of size of community

print()
print("getting CDF of size of community...")
num_communities = max(partition.values()) + 1
print("the number of community:" + str(num_communities))
community_sizes = [0] * num_communities  # 用于存储每个社区的节点数量
for node, community_id in partition.items():
    community_sizes[community_id] += 1
print(max(community_sizes))
print(min(community_sizes))
print(np.mean(community_sizes))
community_list = pd.DataFrame(columns=['size'],data=community_sizes)

# print(size_of_community)
res_freq = stats.relfreq(community_sizes , numbins=100)
cdf_value = np.cumsum(res_freq.frequency)
cdf_value = cdf_value*100
x = res_freq.lowerlimit + np.linspace(0, res_freq.binsize * res_freq.frequency.size, res_freq.frequency.size)
plt.xlabel("size of community")
plt.ylabel("CDF(%)")
# plt.xlim(xmin=0)
plt.ylim(0, 100)
plt.plot(x, cdf_value)
# plt.savefig('CDF_of_size_of_community.png')
plt.show()

# 下面只考虑nonsingleton
nonsingleton = [x for x in community_sizes if x>1]
nonsingleton.sort()
print()
print("getting nonsingleton users...")
nonsingleton_count = len(nonsingleton)
print("nonsingleton_count" + str(nonsingleton_count))
nonsingleton_avg_size = np.average(nonsingleton)
print("avg size:" + str(nonsingleton_avg_size))
nonsingleton_user_count = sum(nonsingleton)
print("nonsingleton user count:" + str(nonsingleton_user_count))
