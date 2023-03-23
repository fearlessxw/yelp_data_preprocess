"""
Sampling: Only consider 10% users with full information
"""


import pandas as pd
import datetime
import easygraph as eg

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# load the dataset
print("loading the dataset...")
path = '../yelp_dataset/'
users = pd.read_json(path + 'yelp_academic_dataset_user.json', orient='columns', lines=True)
print("loading users succeed...")
users = users[["user_id", "friends"]]
users = users.sample(frac=0.1, replace=False, axis=0, random_state=None)
total_users = users.shape[0]
print(users.shape)


# print("sampling...")
# sub_users = users.sample(frac=0.1, replace=False, axis=0, random_state=None)
G = eg.Graph()
G.add_nodes_from(list(users['user_id']))
# print(sub_users.shape)


# add edges
pre_time = datetime.datetime.now()
edge_list = []


def relationship(row):
    if row['friends'] != 'None':
        edge_list.extend([(row['user_id'], x.strip()) for x in str(row['friends']).split(',')])


pre_time = datetime.datetime.now()
users.apply(relationship, axis=1)
print("size of edge_list:" + str(len(edge_list)))
cur_time = datetime.datetime.now()
print("time:" + str(cur_time-pre_time))


print("getting the graph...")
pre_time = datetime.datetime.now()
print(len(edge_list))
G.add_edges_from(edge_list)
cur_time = datetime.datetime.now()
print("total_users:" + str(G.number_of_nodes()))
print("total_edges:" + str(G.number_of_edges()))
print("time:" + str(cur_time-pre_time))


print("getting subgraph...")
pre_time = datetime.datetime.now()
G = G.nodes_subgraph(list(users['user_id']))
edges = list(G.edges)
print("total_users:" + str(G.number_of_nodes()))
print("total_edges:" + str(G.number_of_edges()))
print("time:" + str(cur_time-pre_time))

edge = pd.DataFrame(data=edges, columns=['user1','user2','weight'])
edge.drop(columns='weight', inplace=True)
edge.to_csv("graph.csv", index=False, sep='\t')