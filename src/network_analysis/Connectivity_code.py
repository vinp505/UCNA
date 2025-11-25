import networkx as nx
from itertools import combinations
G = nx.read_graphml("dataExtracted/THE_GRAPH.graphml")
def metanodes(G, meta_key='meta'):
    return [n for n ,d in G.nodes(data=True) if d.get(meta_key) is True ]

def widest_path_all_pairs(G,cap_key='capacity', meta_key='meta'):
    T=nx.maximum_spanning_tree(G, weight=cap_key)
    for u, v in T.edges():
      T[u][v]["traffic"] = 0.0

    ms=metanodes(G, meta_key)
    ms=sorted(ms, key=str)
    pair_widest={}
    for a, b in combinations(ms,2):
       try:
        path=nx.shortest_path(T,a,b)
        #min capacity on the path
        bottleneck= min(T[u][v][cap_key] for u,v in zip(path[:-1],path[1:]))
        for u, v in zip(path[:-1], path[1:]):
             T[u][v]["traffic"] += bottleneck
        pair_widest[(a, b)] = {"capacity": float(bottleneck), "path": path}
       except (nx.NetworkXNoPath, nx.NodeNotFound):
          pair_widest[(a, b)]={'capacity':0.0, 'Path':None}
       
       #Average per metanode
    avg_country={}
    for a in ms:
          vals=[]
          for x,y in pair_widest:
             if a==x or a==y:
                vals.append(pair_widest[(x,y)]['capacity'])
          avg_country[a]=(sum(vals) / len(vals)) if vals else 0.0
       #Most important edge
    max_edge = max(T.edges(), key=lambda e: T[e[0]][e[1]]['traffic'])
    max_value = T[max_edge[0]][max_edge[1]]['traffic']
    return pair_widest,avg_country,  max_edge , max_value

pair_widest, avg_country, most_important_edge ,value= widest_path_all_pairs(G)

# Example: print top-10 countries by average widest-path capacity
top10 = sorted(avg_country.items(), key=lambda kv: kv[1], reverse=True)[:10]
for node, score in top10:
    ctry = G.nodes[node].get("country", "unknown")
    print(f"{ctry} ({node}): {score:.2f} Gbps")

# Example: one pair (a,b)
# a, b = next(iter(pair_widest))
# info = pair_widest[(a, b)]
# print("Pair:", a, b, "capacity:", info["capacity"], "path:", info["path"])

print(most_important_edge)
print(value)

