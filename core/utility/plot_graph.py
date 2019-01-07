#!/usr/bin/env python
# -*- coding: utf-8 -*

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import numpy as np
import math
import os
import pandas as pd

from sklearn.metrics import euclidean_distances
from scipy.sparse.csgraph import minimum_spanning_tree

import plotly.plotly as py
import plotly.offline as pyoff
from plotly.graph_objs import *
import networkx as nx


import plotly.graph_objs as go
import io

from utilities import *

import core.configurations
import core.utility.logger as logger

conf = core.configurations.get_conf()
log_path = conf.get('MAIN', 'log_path')
log_file_name = conf.get('MAIN', 'log_file_name')
log = logger.getLogger(__name__, log_path, log_file_name)

def plot_similarity_graph(X, words_list, file_name, type_chart, node_style = "text"):

    try:
        similarities = euclidean_distances(X)

        min_graph = minimum_spanning_tree(csgraph=similarities, overwrite=False)

        M_min_graph = min_graph.toarray()

        ceil_matrix = np.vectorize(lambda x: 1 if x > 0 else 0)

        adjacency_matrix = ceil_matrix(M_min_graph)

        G = create_graph(adjacency_matrix, graph_layout = 'fruchterman_reingold')

        if type_chart == "json" or type_chart == "d3":
            json_file = createJsonD3Graph(G, words_list)
            url = creteNetworkD3Graph(json_file, file_name)
        else:
            node_trace, edge_trace = creteEdges(G, words_list, node_style)
            node_trace = colorNodesPoints(G, node_trace)
            url = creteNetworkGraph(edge_trace, node_trace, file_name)

        if type_chart == "json":
            return json_file

        log.info("MST built in " + file_name)
        return url
    except Exception as e:
        log.error("Impossible MST building")
        log.error("Unexpected error:", e)
        return None

# write json for d3 graph
def createJsonD3Graph(G, words_list):
    from networkx.readwrite import json_graph
    import json

    data = json_graph.node_link_data(G)
    list = data["nodes"]
    for el in list:
        el["pos"] = el["pos"].tolist()  # convert numpy array to list
        el['name'] = words_list[el["id"]]  # add name of node

    # with open(file_name, 'w') as f:
    #     json.dump(data, f, indent=4)

    return data



def creteNetworkD3Graph(json_file, file_name):
    import json
    html = '<!DOCTYPE html>'+\
           '<meta charset="utf-8">'+\
           '<style>'+\
            ' .link {'+\
            '  stroke: #000;'+\
            '  stroke-width: 1.5px;'+\
            '}'+\
            ' .node {'+\
            '  cursor: move;'+\
            '  fill: #ccc;'+\
            '  stroke: #000;'+\
            '  stroke-width: 1.5px;'+\
            '}'+\
            ' .node.fixed {'+\
            '  fill: #f00;'+\
            '}'+\
            '</style>'+\
            '<body>'+\
            '<script src="https://d3js.org/d3.v3.min.js"></script>'+\
            '<script>'+\
            ' var width = 2120,'+\
            ' height = 1580;'+\
            'var force = d3.layout.force()'+\
            '    .size([width, height])'+\
            '    .charge(-400)'+\
            '    .linkDistance(40)'+\
            '    .on("tick", tick);'+\
            'var drag = force.drag()'+\
            '    .on("dragstart", dragstart);'+\
            'var svg = d3.select("body").append("svg")'+\
            '    .attr("width", width)'+\
            '    .attr("height", height);'+\
            'var link = svg.selectAll(".link"),'+\
            '    node = svg.selectAll(".node");'+\
            'graph = '+json.dumps(json_file)+';'+\
            ' force'+\
            '  .nodes(graph.nodes)'+\
            '  .links(graph.links)'+\
            '  .start();'+\
            'link = link.data(graph.links)'+\
            '.enter().append("line")'+\
            '  .attr("class", "link");'+\
            ' node1 = node.data(graph.nodes)'+\
            '.enter().append("circle")'+\
            '  .attr("class", "node")'+\
            '  .attr("r", 8)'+\
            '  .on("dblclick", dblclick)'+\
            '  .call(drag);'+\
            'node2 = node.data(graph.nodes)'+\
            '.enter().append("text")'+\
            '  .attr("class", "node")'+\
            '  .attr("dx", function(d) { return d.x })'+\
            '  .attr("dy", function(d) { return d.y })'+\
            '  .html(function(d) { return d.name });'+\
            'function tick() {'+\
            '  link.attr("x1", function(d) { return d.source.x; })'+\
            '      .attr("y1", function(d) { return d.source.y; })'+\
            '      .attr("x2", function(d) { return d.target.x; })'+\
            '      .attr("y2", function(d) { return d.target.y; });'+\
            '  node1.attr("cx", function(d) { return d.x; })'+\
            '      .attr("cy", function(d) { return d.y; });'+\
            '  node2.attr("dx", function(d) { return d.x+15; })'+\
            '      .attr("dy", function(d) { return d.y; });'+\
            '}'+\
            'function dblclick(d) {'+\
            '  d3.select(this).classed("fixed", d.fixed = false);'+\
            '}'+\
            'function dragstart(d) {'+\
            '  d3.select(this).classed("fixed", d.fixed = true);'+\
            '}'+\
            '</script>'

    with open(file_name, 'w') as f:
        f.write(html)
    return file_name


def create_graph(adjacency_matrix, graph_layout = 'shell'):
    # given an adjacency matrix use networkx and matlpotlib to plot the graph

    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)

    # these are different layouts for the network you may try
    # shell seems to work best
    if graph_layout == 'spring':
        graph_pos = nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos = nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos = nx.random_layout(G)
    elif graph_layout == 'kk':
        graph_pos = nx.kamada_kawai_layout(G)
    elif graph_layout == 'fruchterman_reingold':
        graph_pos = nx.fruchterman_reingold_layout(G)
    else:
        graph_pos = nx.shell_layout(G)

    nx.set_node_attributes(G, name='pos', values=graph_pos)

    return G


def creteEdges(G, words_list, node_style):

    edge_trace = Scatter(
        x=[],
        y=[],
        line=Line(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode = node_style,#'text',#'markers'
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)
        node_trace['text'].append(words_list[node])

    return node_trace, edge_trace

def colorNodesPoints(G, node_trace):

    # for node, adjacencies in enumerate(G.adjacency_list()):
    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'].append(len(adjacencies))
        node_info = '# of connections: ' + str(len(adjacencies))
        node_trace['text'].append(node_info)

    return node_trace


def creteNetworkGraph(edge_trace, node_trace, filename):

    name_figure = 'Similarity Graph for ' + os.path.basename(filename).split('.')[0].split('_')[-1]

    fig = Figure(data=Data([edge_trace, node_trace]),
                 layout=Layout(
                     title='<br>' + name_figure,
                     titlefont=dict(size=16),
                     showlegend=False,
                     hovermode='closest',
                     margin=dict(b=20, l=5, r=5, t=40),
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    # py.iplot(fig, filename=filename)

    return pyoff.offline.plot(fig, filename=filename, auto_open=False)

# def createPajakNetworkGraph(G, filename):
#     nx.write_pajek(G, filename+".net")

def plotFrequencyGraph(cell_frequency, codebook2word, filename, num=0, type='histogram'):

    widht = len(cell_frequency)

    # CREATING DATAFRAME
    a = np.array(cell_frequency)
    a = a.transpose()
    frequencies = [item for sublist in a.tolist() for item in sublist]
    index = range(len(frequencies))

    # create dictionary with list of different lenght (with NaN)
    d = dict(index=np.array(index), frequency=np.array(frequencies), word=np.array(codebook2word))
    frequency_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.iteritems()]))
    # removing NaN rows
    frequency_df = frequency_df[frequency_df.word != '']

    # sorting by frequency
    frequency_df = frequency_df.sort_values(['frequency'], ascending=False)

    lenght = len(frequency_df)
    if num > lenght or num == 0:
        num = lenght

    color = []
    area = []
    words = []
    x = []
    y = []

    for i in range(0,num):
        word = frequency_df.iloc[i]['word']
        if (word != ''):
            x_i = frequency_df.iloc[i]['index']/widht+1
            x.append(x_i)
            y_i = frequency_df.iloc[i]['index']%widht+1
            y.append(y_i)
            color.append(randomColor())
            f = frequency_df.iloc[i]['frequency']
            area.append(f)
            words.append(word)


    if type == 'bubble':
        plotBubbleChart(x, y, words, color, area, filename)
    else:
        plotBarChart(words, area, filename)

def plotBubbleChart(x, y, text, color, area, filename):

    new_area = map(lambda x: x * 1500, area)
    trace0 = go.Scatter(
        x=x,
        y=y,
        text=text,
        mode='markers+text',
        marker=dict(
            color=color,
            size=new_area,
        ),
        textposition='top center',
        hoverinfo='none'
    )

    data = go.Data([trace0])
    return pyoff.offline.plot(data, filename=filename, auto_open=False)

def plotBarChart(text, values, filename):

    data = [go.Bar(
        x=values,
        y=text,
        orientation='h',
        hoverinfo = 'none'
    )]

    layout = go.Layout(
        xaxis=dict(
            # autorange=True,
            # showgrid=False,
            # zeroline=False,
            # showline=False,
            # autotick=True,
            # ticks='',
            # showticklabels=False
        )#,
        # yaxis=dict(
        #     autorange=True,
        #     showgrid=False,
        #     zeroline=False,
        #     showline=False,
        #     autotick=True,
        #     ticks='',
        #     showticklabels=False
        # )
    )

    fig = go.Figure(data=data, layout=layout)
    pyoff.offline.plot(fig, filename=filename, auto_open=False)

    # pyoff.offline.plot(data, filename=filename)

def plotMatrix(matrix, filename):
    # plt.matshow(matrix)
    plt.imshow(matrix, interpolation="none")
    plt.savefig(filename)
