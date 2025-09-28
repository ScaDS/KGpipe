# Plotting Pipelines

from kgpipe.common import KgPipe, KgTask, Data, DataFormat
from dataclasses import dataclass
from pathlib import Path
import json
from kgpipe.common.models import KG
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch

def plot_pipeline(build_json):
    """
    Plot the pipeline from build JSON showing tasks as nodes and I/O as edges.
    
    Args:
        build_json: The build JSON from pipeline.build()
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes for each task
    for i, task_info in enumerate(build_json):
        task_name = task_info['task']
        task_id = f"task_{i}"
        G.add_node(task_id, label=task_name, type='task')
    
    # Add source and target nodes
    G.add_node("source", label="Source RDF", type='data')
    G.add_node("target", label="Target RDF", type='data')
    G.add_node("result", label="Result RDF", type='data')
    
    # Add edges based on data flow
    for i, task_info in enumerate(build_json):
        task_id = f"task_{i}"
        
        # Connect inputs to this task
        for input_data in task_info.get('input', []):
            if "source.nt" in str(input_data.path):
                G.add_edge("source", task_id, label="source")
            elif "seed.nt" in str(input_data.path):
                G.add_edge("target", task_id, label="target")
            else:
                # This is an intermediate output from a previous task
                for j in range(i):
                    prev_task_id = f"task_{j}"
                    G.add_edge(prev_task_id, task_id, label=f"output_{j}")
        
        # Connect outputs from this task
        for output_data in task_info.get('output', []):
            if "result.nt" in str(output_data.path):
                G.add_edge(task_id, "result", label="final_output")
            else:
                # This output goes to a subsequent task
                for j in range(i + 1, len(build_json)):
                    next_task_id = f"task_{j}"
                    G.add_edge(task_id, next_task_id, label=f"output_{i}")
    
    # Create the plot
    plt.figure(figsize=(16, 8))
    
    # Create left-to-right layout manually
    pos = {}
    
    # Position source data on the left
    pos["source"] = (0, 0.5)
    pos["target"] = (0, -0.5)
    
    # Position tasks in order from left to right
    task_nodes = [n for n in G.nodes() if n.startswith('task_')]
    for i, task_node in enumerate(task_nodes):
        x_pos = (i + 1) * 2.0  # Spread tasks horizontally
        y_pos = 0
        pos[task_node] = (x_pos, y_pos)
    
    # Position result on the right
    if task_nodes:
        max_x = max(pos[node][0] for node in pos if node != "result")
        pos["result"] = (max_x + 2, 0)
    else:
        pos["result"] = (2, 0)
    
    # Draw nodes with different colors based on type
    task_nodes = [n for n in G.nodes() if n.startswith('task_')]
    data_nodes = [n for n in G.nodes() if not n.startswith('task_')]
    
    # Draw task nodes in blue
    if task_nodes:
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=task_nodes,
                              node_color='lightblue',
                              node_size=4000)
    
    # Draw data nodes in green
    if data_nodes:
        nx.draw_networkx_nodes(G, pos, 
                              nodelist=data_nodes,
                              node_color='lightgreen',
                              node_size=4000)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->',
                          connectionstyle='arc3,rad=0.1')
    
    # Add labels
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    
    # Add edge labels (simplified to avoid clutter)
    edge_labels = {}
    for u, v, d in G.edges(data=True):
        if d.get('label') in ['source', 'target', 'final_output']:
            edge_labels[(u, v)] = d['label']
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title("Pipeline Execution Flow", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return plt