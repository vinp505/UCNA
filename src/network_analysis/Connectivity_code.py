import networkx as nx
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imageio
import seaborn as sns
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent.parent#obtain directory of this file
_PROJ_DIR = _FILE_DIR.parent#obtain main project directory
_DATA_DIR = _PROJ_DIR / "dataset"
_DATAEXTR_DIR = _PROJ_DIR / "dataExtracted"
_EXTRACT_DIR = _PROJ_DIR / "dataExtracted"
_VISUAL_DIR = _PROJ_DIR / "visualizations"

G = nx.read_graphml(str(_EXTRACT_DIR / "THE_GRAPH.graphml"))
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
    return pair_widest, avg_country,  max_edge , max_value

pair_widest, avg_country, most_important_edge ,value= widest_path_all_pairs(G)

def average_ping(G, meta_key ='meta'):
    for u, v in G.edges():
        G[u][v]["Count"] = 0.0
    ms=metanodes(G, meta_key)
    ms=sorted(ms, key=str)
    pair_length_dict={}
    for a,b in combinations(ms,2):
        length=nx.shortest_path_length(G,a,b)
        path=nx.shortest_path(G,a,b)
        for u, v in zip(path[:-1], path[1:]):
             G[u][v]["Count"] += 1
        pair_length_dict[(a, b)] = {"length": float(length)}
    avg_length={}
    for a in ms:
        vals=[]

        for (x, y), data in pair_length_dict.items():
            if x == a:
                vals.append(data['length'])
                avg_length[a]=(sum(vals) / len(vals)) if vals else 0.0
    fastest_ping_metanode = min(avg_length.items(), key=lambda x: x[1])[0]
    avg_length_fast=min(avg_length.items(), key=lambda x: x[1])[1]
    most_important_edge= max(G.edges(), key=lambda e: G[e[0]][e[1]]['Count'])
    max_value = G[most_important_edge[0]][most_important_edge[1]]['Count']
    return fastest_ping_metanode, avg_length_fast, most_important_edge,max_value



# Example: print top-10 countries by average widest-path capacity
# top10 = sorted(avg_country.items(), key=lambda kv: kv[1], reverse=True)[:10]
# for node, score in top10:
#     ctry = G.nodes[node].get("country", "unknown")
#     print(f"{ctry} ({node}): {score:.2f} Gbps")
fastest_metanode, avg_length, mvp_edge, count=average_ping(G)
ctr_name= G.nodes[fastest_metanode].get('country')
print(ctr_name,avg_length,mvp_edge,count)



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

def get_top10(avgConnectivityDf:pd.DataFrame, rankDf:pd.DataFrame, n: int, from_start= False):
    
    # set start of the slicing
    if from_start:
        start = 0
    else:
        start = n-5
    
    # only keep countries that are in the top10 at any point
    for col in avgConnectivityDf.columns:
        if min(rankDf[col].values) > 10:
            rankDf.drop(col, axis= 1, inplace= True)
            avgConnectivityDf.drop(col, axis= 1, inplace= True)

    # only get data for chosen period
    idx = list(range(start, n))
    conn_df = pd.DataFrame(avgConnectivityDf.loc[idx])

    # sort by the last relevant iteration so that the legend is in order
    return conn_df.T.sort_values(by= n-1, axis= 0, ascending= False)

def visualizeResults(avgConnectivityDf:pd.DataFrame, rankDf:pd.DataFrame, from_start= False):
    
    # list to store frames, one frame per slider value
    frames = []
    slider_values = range(5, len(avgConnectivityDf)+1)

    # map country to color
    columns = avgConnectivityDf.columns
    colors = sns.color_palette("turbo", len(columns))

    col_dict = {columns[i] : colors[i] for i in range(len(columns))}

    # iterate through slider values
    for n in slider_values:

        # new plot each time
        fig, ax = plt.subplots(figsize= (10, 7))
        
        # get entries    
        conn_df = get_top10(avgConnectivityDf, rankDf, n, from_start)
        
        # set start
        if from_start:
            start = 0
        else:
            start = n-5

        ax.grid(alpha= 0.3)
        
        # keep track of minimum connectivity to adjust graph lines
        min_pos_conn = np.inf
        
        # plot lines
        idx = list(range(start, n))
        for i, (_, r) in enumerate(conn_df.iterrows()):
            
            # use top10 of final iteration to label countries
            if i < 10:
                min_conn_row = r.values.min()
                if (min_conn_row <= min_pos_conn):
                    min_pos_conn = min_conn_row
                l = f"{i+1}. {r.name}"
            
            # no label for countries outside of the top10
            else:
                l = None

            ax.plot(list(range(len(idx))), r.values, marker= 'o', label= l, c= col_dict[r.name])
        
        ax.set_xticks(list(range(len(idx))), idx)
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
    if from_start:
        name = "connectivity_top10_fromstart.gif"
    else:
        name = "connectivity_top10_slices.gif"

    imageio.mimsave(str(_VISUAL_DIR / name), frames, fps=2)

def simulateAttacks(G: nx.Graph):
    #get the average connectivity of each country and the first most important edge
    _, avg_countryInit, maxEdge, _ = widest_path_all_pairs(G)
    avgConnectivities = []
    iteration = 1
    already_dropped = set()
    while len(already_dropped) < len(avg_countryInit):#run until every country has dropped out
        G.remove_edge(maxEdge[0], maxEdge[1])#remove most important edge
        _, avg_country, maxEdge, _ = widest_path_all_pairs(G)#repeat
        #now check which countries dropped out and write that to a log file
        already_dropped = check_drop_outs(avg_countryInit, avg_country, iteration=iteration, threshold=0.5, already_dropped=already_dropped)
        avgConnectivities.append(avg_country)
        iteration += 1
    avgConnectivityDf = pd.DataFrame(avgConnectivities, dtype=float)#create dataframe with average connectivities of countries for each iteration (countries are columns, so a row represents the average connectivities for all countries at that iteration)
    rankDf = avgConnectivityDf.rank(axis=1, ascending=False, method="first").copy()#handle ties randomly (order of array)
    avgConnectivityDf.to_csv(str(_DATAEXTR_DIR / "avgConnectivity.csv"), index= False)
    rankDf.to_csv(str(_DATAEXTR_DIR / "rank.csv"), index= False)
    visualizeResults(avgConnectivityDf, rankDf)