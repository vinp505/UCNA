import networkx as nx
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import seaborn as sns
from typing import Literal
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "dataset"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"
_VISUAL_DIR = _PROJ_DIR / "visualizations"

G = nx.read_graphml(str(_EXTRACT_DIR / "THE_MERGED_GRAPH.graphml"))

def metanodes(G, meta_key='meta'):
    return [n for n ,d in G.nodes(data=True) if d.get(meta_key) is True ]

def widest_path_all_pairs(G: nx.MultiGraph, cap_key='capacity', meta_key='meta'):
    mst=nx.maximum_spanning_edges(G, weight=cap_key)

    edgelist = list(mst)

    edge_key_dict = {}
    for u, v, k, _ in edgelist:
        edge_key_dict[(u, v)] = k
        edge_key_dict[(v, u)] = k
    subgraph = [(u, v, k) for u, v, k, _ in edgelist]
    T = G.edge_subgraph(subgraph).copy()
    for u, v, k in T.edges(keys= True):
      T[u][v][k]["traffic"] = 0.0

    ms=metanodes(G, meta_key)
    ms=sorted(ms, key=str)
    pair_widest={}
    for a, b in combinations(ms,2):
       try:
        path=nx.shortest_path(T,a,b)
        #min capacity on the path
        bottleneck= min(T[u][v][edge_key_dict[(u, v)]][cap_key] for u,v in zip(path[:-1],path[1:]))
        for u, v in zip(path[:-1], path[1:]):
             T[u][v][edge_key_dict[(u, v)]]["traffic"] += bottleneck
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
    max_edge = max(T.edges(keys= True), key=lambda e: T[e[0]][e[1]][e[2]]['traffic'])
    max_value = T[max_edge[0]][max_edge[1]][max_edge[2]]['traffic']
    return pair_widest, avg_country,  max_edge , max_value

# pair_widest, avg_country, most_important_edge ,value= widest_path_all_pairs(G)

# Example: print top-10 countries by average widest-path capacity
# top10 = sorted(avg_country.items(), key=lambda kv: kv[1], reverse=True)[:10]
# for node, score in top10:
#     ctry = G.nodes[node].get("country", "unknown")
#     print(f"{ctry} ({node}): {score:.2f} Gbps")

# Example: one pair (a,b)
# a, b = next(iter(pair_widest))
# info = pair_widest[(a, b)]
# print("Pair:", a, b, "capacity:", info["capacity"], "path:", info["path"])

# print(most_important_edge)
# print(value)
def check_drop_outs(avg_countryInit:dict, avg_country:dict, iteration: int, threshold: float = 0.5, already_dropped: set = None, logfile: str = str(_EXTRACT_DIR / "dropouts.txt")):
    #Initialize already_dropped set if not provided
    if already_dropped is None:
        already_dropped = set()
    #clear logfile at first iteration  
    if iteration == 1:
        open(logfile, "w").close()
    #log dropouts
    with open(logfile, 'a') as f:
        for country, init_cap in avg_countryInit.items():
            current_cap = avg_country.get(country, 0.0)
            if (current_cap / init_cap) < threshold and country not in already_dropped:
                f.write(f"Iteration {iteration}: Country {country} dropped out (from {init_cap:.2f} to {current_cap:.2f})\n")
                already_dropped.add(country)

    return already_dropped

def get_top10(avgConnectivityDf:pd.DataFrame, n: int, relative= False):
    
    # set start of the slicing
    start = n-30
    
    if relative:
        ConnDf = (avgConnectivityDf / avgConnectivityDf.iloc[0]).copy()
    else:
        ConnDf = avgConnectivityDf.copy()

    # only get data for chosen period
    idx = list(range(start, n))
    conn_df = pd.DataFrame(ConnDf.loc[idx])

    # sort by the last relevant iteration so that the legend is in order
    return conn_df.T.sort_values(by= n-1, axis= 0, ascending= False)

def visualizeResults(avgConnectivityDf:pd.DataFrame, mode, relative= False):
    
    # list to store frames, one frame per slider value
    frames = []
    slider_values = range(30, len(avgConnectivityDf)+1)

    # map country to color
    columns = avgConnectivityDf.columns
    colors = sns.color_palette("turbo", len(columns))

    col_dict = {columns[i] : colors[i] for i in range(len(columns))}

    # iterate through slider values
    for n in slider_values:

        # new plot each time
        fig, ax = plt.subplots(figsize= (15, 7))
        
        # get entries    
        conn_df = get_top10(avgConnectivityDf, n, relative)
        
        # set start
        start = n-30

        ax.grid(alpha= 0.3)
    
        # keep track of minimum connectivity to adjust graph lims
        min_pos_conn = np.inf

        # plot lines
        idx = list(range(start, n))
        for i, (_, r) in enumerate(conn_df.iterrows()):

            # use top10 of final iteration to label countries
            if i < 10:

                # update min conn if needed
                min_conn_row = r.values.min()
                if (min_conn_row <= min_pos_conn):
                    min_pos_conn = min_conn_row

                l = f"{i+1}. {r.name}" 
            
            # no label for countries outside of the top10
            else:
                l = None

            ax.plot(list(range(len(idx))), r.values, marker= 'o', label= l, c= col_dict[r.name])
        

        ax.set_xticks(list(range(len(idx))), idx)

        if relative:
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
            ax.axhline(y= 0.5, linestyle= '--', c= 'darkred') 
        else:
            ax.set_ylim(bottom= min_pos_conn-1)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        ax.set_ylabel("Connectivity")
        ax.set_xlabel("Iteration")
        ax.set_title(f"Top 10 countries by connectivity.\nFrom iteration {start} to {n-1}.")
        
        fig.tight_layout()
 
        # save frame
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # magic ???
        image = np.asarray(renderer.buffer_rgba())[:, :, :3]

        # store frame, close plot
        frames.append(image)
        plt.close(fig)

    # save gif
    if relative:
        name = f"connectivity_top10_{mode}_relative.gif"
    else:
        name = f"connectivity_top10_{mode}_absolute.gif"

    imageio.mimsave(name, frames, fps=2)

def simulateAttacks(G: nx.MultiGraph, mode: Literal['targeted', 'random', 'ping']= "targeted", logfile: str = str(_EXTRACT_DIR / "iterations.txt")):
    
    if mode == 'targeted':
        #get the average connectivity of each country and the first most important edge
        _, avg_countryInit, maxEdge, _ = widest_path_all_pairs(G)
    
    if mode == 'ping':
        pass  # Andrei, add function call here

    if mode == 'random':
        pass  # Greg, add function call here

    avgConnectivities = []
    iteration = 1
    already_dropped = set()
    while len(already_dropped) < len(avg_countryInit):#run until every country has dropped out
        statement = f"Iteration {iteration}: removing edge {maxEdge[0]} <-> {maxEdge[1]}"
        with open(logfile, 'a') as f:
            f.write(f"{statement}\n")
        print(statement)
        G.remove_edge(maxEdge[0], maxEdge[1], key= maxEdge[2])#remove most important edge
        _, avg_country, maxEdge, _ = widest_path_all_pairs(G)#repeat
        #now check which countries dropped out and write that to a log file
        already_dropped = check_drop_outs(avg_countryInit, avg_country, iteration=iteration, threshold=0.5, already_dropped=already_dropped)
        avgConnectivities.append(avg_country)
        iteration += 1
    avgConnectivityDf = pd.DataFrame(avgConnectivities, dtype=float)#create dataframe with average connectivities of countries for each iteration (countries are columns, so a row represents the average connectivities for all countries at that iteration)
    avgConnectivityDf.to_csv(str(_EXTRACT_DIR / f"avgConnectivity_{mode}.csv"), index= False)
    visualizeResults(avgConnectivityDf, mode, relative= False)
    visualizeResults(avgConnectivityDf, mode, relative= True)

# RUN IT
for mode in ['targeted', 'random', 'ping']:
    simulateAttacks(G, mode= mode)