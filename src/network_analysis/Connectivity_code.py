import networkx as nx
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import seaborn as sns
from typing import Literal
from pathlib import Path
import random as rnd

rnd.seed(42)

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "dataset"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"
_VISUAL_DIR = _PROJ_DIR / "visualizations"

G = nx.read_graphml(str(_EXTRACT_DIR / "THE_MERGED_GRAPH.graphml"))

def metanodes(G, meta_key='meta'):
    return [n for n ,d in G.nodes(data=True) if d.get(meta_key, False)]

def metanode_to_country(G, meta_key='meta'):
    return {n : d['country'] for n, d in G.nodes(data=True) if d.get(meta_key, False)}

def widest_path_all_pairs(G: nx.MultiGraph, meta_dict: dict, cap_key='capacity', meta_key='meta'):

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
          avg_country[meta_dict[a]]=(sum(vals) / len(vals)) if vals else 0.0
       #Most important edge
    max_edge = max(T.edges(keys= True), key=lambda e: T[e[0]][e[1]][e[2]]['traffic'])
    max_value = T[max_edge[0]][max_edge[1]][max_edge[2]]['traffic']
    return pair_widest, avg_country,  max_edge , max_value

# pair_widest, avg_country, most_important_edge ,value= widest_path_all_pairs(G)

def average_ping(G, meta_dict: dict, meta_key ='meta'):

    edge_key_dict = {}
    for u, v, k in G.edges(keys= True):
        G[u][v][k]["Count"] = 0.0
        edge_key_dict[(u, v)] = k
        edge_key_dict[(v, u)] = k
    ms=metanodes(G, meta_key)
    ms=sorted(ms, key=str)
    pair_length_dict={}
    for a,b in combinations(ms,2):
        try:
            length=nx.shortest_path_length(G,a,b)*50  # 50ms added for each rerouting
            path=nx.shortest_path(G,a,b)
            for u, v in zip(path[:-1], path[1:]):
                G[u][v][edge_key_dict[(u, v)]]["Count"] += 1
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            length = 600  # not connected: have to use satellites ~600 ms
        pair_length_dict[(a, b)] = {"length": float(length)}
    avg_length={}
    for a in ms:
        vals=[]

        for (x, y), data in pair_length_dict.items():
            if x == a or y == a:
                vals.append(data['length'])
        avg_length[meta_dict[a]]=(sum(vals) / len(vals)) if vals else 0.0
    fastest_ping_metanode = min(avg_length.items(), key=lambda x: x[1])[0]
    avg_length_fast=min(avg_length.items(), key=lambda x: x[1])[1]
    most_important_edge= max(G.edges(keys= True), key=lambda e: G[e[0]][e[1]][e[2]]['Count'])
    max_value = G[most_important_edge[0]][most_important_edge[1]][most_important_edge[2]]['Count']
    return fastest_ping_metanode, avg_length, most_important_edge, max_value



# Example: print top-10 countries by average widest-path capacity
# top10 = sorted(avg_country.items(), key=lambda kv: kv[1], reverse=True)[:10]
# for node, score in top10:
#     ctry = G.nodes[node].get("country", "unknown")
#     print(f"{ctry} ({node}): {score:.2f} Gbps")


# print(most_important_edge)
# print(value)
def check_drop_outs(avg_countryInit:dict, avg_country:dict, iteration: int, mode: str, random: bool, threshold: float = 0.5, already_dropped: set = None):
    rnd = "random" if random else "targeted"
    logfile = str(_EXTRACT_DIR / f"dropouts_{mode}_{rnd}.txt")

    if mode == 'ping':
        threshold = 1 / threshold
    
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
            if (current_cap / init_cap) < threshold and country not in already_dropped and mode == 'connectivity':
                f.write(f"Iteration {iteration}: Country {country} dropped out (from {init_cap:.2f} to {current_cap:.2f})\n")
                already_dropped.add(country)
            elif ((current_cap / init_cap) > threshold or current_cap >= 450) and country not in already_dropped and mode == 'ping':
                f.write(f"Iteration {iteration}: Country {country} dropped out (from {init_cap:.2f} to {current_cap:.2f})\n")
                already_dropped.add(country)

    return already_dropped

def pick_random_edge(G: nx.MultiGraph):

    return rnd.choice(list(G.edges(keys=True)))

def get_top10(avgConnectivityDf:pd.DataFrame, n: int, mode: str, relative= False):
    
    # set start of the slicing
    start = n-30
    
    if relative:
        ConnDf = (avgConnectivityDf / avgConnectivityDf.iloc[0]).copy()
    else:
        ConnDf = avgConnectivityDf.copy()

    # only get data for chosen period
    idx = list(range(start, n))
    conn_df = pd.DataFrame(ConnDf.loc[idx])

    asc = False if mode == 'connectivity' else True
    
    # sort by the last relevant iteration so that the legend is in order
    return conn_df.T.sort_values(by= n-1, axis= 0, ascending= asc)

def visualizeResults(avgConnectivityDf:pd.DataFrame, mode: str, random: bool, relative= False):
    
    rand = "random" if random else "targeted"
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

                l = f"{i+1}. {r.name} ({round(r.values[-1], 3)})" 
            
            # no label for countries outside of the top10
            else:
                l = None

            ax.plot(list(range(len(idx))), r.values, marker= 'o', label= l, c= col_dict[r.name])
        

        ax.set_xticks(list(range(len(idx))), idx)

        if relative and mode == 'connectivity':
            ax.set_ylim(-0.1, 1.1)
            ax.set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
            ax.axhline(y= 0.5, linestyle= '--', c= 'darkred') 
        
        elif relative and mode == 'ping':
            ax.set_ylim(0.9, 3.1)
            ax.set_yticks([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
            ax.axhline(y= 2, linestyle= '--', c= 'darkred') 

        else:
            ax.set_ylim(bottom= min_pos_conn-1)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        ax.set_ylabel(f"{mode}")
        ax.set_xlabel("Iteration")
        ax.set_title(f"Top 10 countries by {mode}.\nFrom iteration {start} to {n-1}.")
        
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
        name = f"{mode}_top10_{rand}_relative.gif"
    else:
        name = f"{mode}_top10_{rand}_absolute.gif"

    imageio.mimsave(name, frames, fps=2)

def simulateAttacks(real_G: nx.MultiGraph, mode: Literal['connectivity', 'ping']= "targeted", random: bool= False):
    
    G = real_G.copy()
    meta_dict = metanode_to_country(G)

    rnd = "random" if random else "targeted"
    logfile = str(_EXTRACT_DIR / f"iterations_{mode}_{rnd}.txt")
    open(logfile, "w").close()  # clear file at the beginning
    func_dict = {
        'connectivity' : widest_path_all_pairs,
        'ping' : average_ping,
    }
    func = func_dict[mode]
    
    #get the average connectivity of each country and the first most important edge
    _, avg_countryInit, maxEdge, _ = func(G, meta_dict)

    avgConnectivities = []
    iteration = 1
    already_dropped = set()
    while len(already_dropped) < len(avg_countryInit):#run until every country has dropped out


        if random:
            # random attack: choose a random edge each iteration
            maxEdge = pick_random_edge(G)
        
        endpoint_1 = meta_dict.get(maxEdge[0], maxEdge[0])
        endpoint_2 = meta_dict.get(maxEdge[1], maxEdge[1])

        statement = (f"Iteration {iteration}: {'randomly' * random} removing edge "
                         f"{endpoint_1} <-> {endpoint_2} (key={maxEdge[2]})")
            
        with open(logfile, 'a') as f:
            f.write(f"{statement}\n")

        print(statement)
        G.remove_edge(maxEdge[0], maxEdge[1], key= maxEdge[2])#remove most important edge
        _, avg_country, maxEdge, _ = func(G, meta_dict)#repeat
        #now check which countries dropped out and write that to a log file
        already_dropped = check_drop_outs(avg_countryInit, avg_country, iteration=iteration, mode= mode, random= random, threshold=0.5, already_dropped=already_dropped)
        avgConnectivities.append(avg_country)
        iteration += 1
        print(f"{len(avg_countryInit) - len(already_dropped)} countries are STILL in.")
    print("End of simulation, saving and plotting results.")
    avgConnectivityDf = pd.DataFrame(avgConnectivities, dtype=float)#create dataframe with average connectivities of countries for each iteration (countries are columns, so a row represents the average connectivities for all countries at that iteration)
    avgConnectivityDf.to_csv(str(_EXTRACT_DIR / f"avg_{mode}_{rnd}.csv"), index= False)
    print("Results saved.")
    visualizeResults(avgConnectivityDf, mode, random, relative= False)
    print("Plot 1/2 saved.")
    visualizeResults(avgConnectivityDf, mode, random, relative= True)
    print("Plot 2/2 saved.")

# RUN IT
for mode in ['connectivity', 'ping']:
    for random in [False, True]:
        
        if mode == 'connectivity':
            continue

        print(f"\n\nRunning {mode} mode, {'random' * random}{'targeted' * (not random)} attacks.")

        simulateAttacks(G, mode= mode, random= random)