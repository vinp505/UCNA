import networkx as nx
from itertools import combinations
import pandas as pd

from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "dataset"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"

G = nx.read_graphml(str(_EXTRACT_DIR / "THE_GRAPH.graphml"))
def metanodes(G, meta_key='meta'):
    return [n for n ,d in G.nodes(data=True) if d.get(meta_key) is True ]

def widest_path_all_pairs(G,cap_key='capacity', meta_key='meta'):
    T=nx.maximum_spanning_tree(G, weight=cap_key)
    ms=metanodes(G, meta_key)
    ms=sorted(ms, key=str)
    pair_widest={}
    for a, b in combinations(ms,2):
       try:
        path=nx.shortest_path(T,a,b)
        #min capacity on the path
        bottleneck= min(T[u][v][cap_key] for u,v in zip(path[:-1],path[1:]))
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
    return pair_widest,avg_country

pair_widest, avg_country = widest_path_all_pairs(G)

# Example: print top-10 countries by average widest-path capacity
top10 = sorted(avg_country.items(), key=lambda kv: kv[1], reverse=True)[:10]
for node, score in top10:
    ctry = G.nodes[node].get("country", "unknown")
    print(f"{ctry} ({node}): {score:.2f} Gbps")

# Example: one pair (a,b)
a, b = next(iter(pair_widest))
info = pair_widest[(a, b)]
print("Pair:", a, b, "capacity:", info["capacity"], "path:", info["path"])

def check_drop_outs(avg_countryInit:dict, avg_country:dict):
    #Greg
    return

def visualizeResults(avgConnectivityDf:pd.DataFrame, rankDf:pd.DataFrame):
    #Vincenzo
    return

def simulateAttacks(G: nx.Graph):
    #get the average connectivity of each country and the first most important edge
    _, avg_countryInit, mostImportantEdge = widest_path_all_pairs(G)
    avgConnectivities = []
    while True:
        G.remove_edge(mostImportantEdge)#remove most important edge
        _, avg_country, mostImportantEdge = widest_path_all_pairs(G)#repeat
        #now check which countries dropped out and write that to a log file
        check_drop_outs(avg_countryInit, avg_country)
        avgConnectivities.append(avg_country)
        break
    avgConnectivityDf = pd.DataFrame(avgConnectivities, dtype=float)#create dataframe with average connectivities of countries for each iteration (countries are columns, so a row represents the average connectivities for all countries at that iteration)
    rankDf = avgConnectivityDf.rank(axis=1, ascending=False, method="first").copy()#handle ties randomly (order of array)
    visualizeResults(avgConnectivityDf, rankDf)