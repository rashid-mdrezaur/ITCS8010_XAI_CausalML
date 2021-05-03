#!/usr/bin/env python3

import json
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt


def main():
    model_graph = nx.DiGraph()
    model_name = 'model_full'
    trial = 19

    params = torch.load(f'models/{model_name}/{model_name}_{trial:02d}.pt')

    hidden_weights = torch.cat([params['hidden_layer.weight'], params['hidden_layer.bias'].unsqueeze(-1)], dim=-1)
    out_weights = torch.cat([params['out_layer.weight'], params['out_layer.bias'].unsqueeze(-1)], dim=-1)

    # print(f'hidden layer: {hidden_weights.shape}')
    # print(f'out layer: {out_weights.shape}')

    in_dim = hidden_weights.shape[-1]  # Includes bias dimension
    hidden_in_dim = hidden_weights.shape[0]
    hidden_out_dim = out_weights.shape[-1]  # Includes bias dimension
    out_dim = out_weights.shape[0]

    in_nodes = make_nodes('in', in_dim)
    hidden_nodes = make_nodes('hidden', hidden_out_dim)
    out_nodes = make_nodes('out', out_dim, has_bias=False)

    model_graph.add_nodes_from(in_nodes)
    model_graph.add_nodes_from(hidden_nodes)
    model_graph.add_nodes_from(out_nodes)

    in_hidden_edges = make_weighted_edges(in_nodes, hidden_nodes[:-1], hidden_weights, normalize=False)
    hidden_out_edges = make_weighted_edges(hidden_nodes, out_nodes, out_weights, normalize=False)

    model_graph.add_edges_from(in_hidden_edges)
    model_graph.add_edges_from(hidden_out_edges)

    # print(nx.dag_longest_path(model_graph, weight='weight'))
    # node = 'hidden_7'
    # for pred in model_graph.predecessors(node):
    #     orig_weight = model_graph[pred][node]['weight'] * model_graph[pred][node]['sign']
    #     print(f'{pred}, {node}: {orig_weight}')

    node_degree = dict(model_graph.degree(weight='weight'))
    node_in_degree = dict(model_graph.in_degree(weight='weight'))
    node_out_degree = dict(model_graph.out_degree(weight='weight'))
    closeness = dict(nx.closeness_centrality(model_graph, distance='distance'))
    rev_closeness = dict(nx.closeness_centrality(model_graph.reverse(), distance='distance'))
    betweenness = dict(nx.betweenness_centrality(model_graph, weight='distance'))

    with open('nn_metrics/node_weighted_degree.json', 'w') as f:
        json.dump(node_degree, f, indent=2)
    with open('nn_metrics/node_in_degree.json', 'w') as f:
        json.dump(node_in_degree, f, indent=2)
    with open('nn_metrics/node_out_degree.json', 'w') as f:
        json.dump(node_out_degree, f, indent=2)
    with open('nn_metrics/node_closeness.json', 'w') as f:
        json.dump(closeness, f, indent=2)
    with open('nn_metrics/node_rev_closeness.json', 'w') as f:
        json.dump(rev_closeness, f, indent=2)
    with open('nn_metrics/node_betweenness.json', 'w') as f:
        json.dump(betweenness, f, indent=2)

    fig = draw_model_graph(model_graph, in_nodes, hidden_nodes, out_nodes)
    plt.savefig('model_graph.png', dpi=300)
    plt.show()


def make_nodes(layer, dim, has_bias=True):
    return [
        f'{layer}_bias' if i == dim-1 and has_bias else f'{layer}_{i}'
        for i in range(dim)
    ]


def make_weighted_edges(in_nodes, out_nodes, weights, normalize=False):
    weights = weights.T
    if normalize:
        weights = F.normalize(weights, dim=-1)
    edges = []
    for i, in_node in enumerate(in_nodes):
        for j, out_node in enumerate(out_nodes):
            edge_weight = weights[i][j].item()
            sign = 1 if edge_weight >= 0 else -1
            edge_weight = abs(edge_weight)
            edges.append((in_node, out_node, {'weight': edge_weight, 'distance': 1/edge_weight, 'sign': sign}))

    return edges


def tripartite_draw_pos(in_nodes, hidden_nodes, out_nodes):
    pos = {}
    pos.update( (n, (1, i)) for i, n in enumerate(in_nodes[::-1]) )
    pos.update((n, (2, i*2+10)) for i, n in enumerate(hidden_nodes[::-1]))
    pos.update((n, (3, i*3+15)) for i, n in enumerate(out_nodes[::-1]))

    return pos


def draw_model_graph(model_graph, in_nodes, hidden_nodes, out_nodes):
    pos = tripartite_draw_pos(in_nodes, hidden_nodes, out_nodes)
    edge_colors = ['k' if model_graph[u][v]['sign'] == 1 else 'r' for u, v in model_graph.edges()]
    edge_widths = [model_graph[u][v]['weight'] * 4 for u, v in model_graph.edges()]

    node_degree = [w * 100 for n, w in model_graph.degree(weight='weight')]

    fig = plt.figure(figsize=(12, 8))
    nx.draw(model_graph, pos=pos, width=edge_widths, edge_color=edge_colors, node_size=node_degree)

    return fig


if __name__ == '__main__':
    main()
