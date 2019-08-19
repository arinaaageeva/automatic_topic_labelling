import pandas as pd

import colorlover as cl
import plotly.graph_objects as go

from labeler import *

def get_colors(parent_colors_map, values):
    
    colors_map = {}
    for parent, color in parent_colors_map.items():
        
        current_values = values[values.iloc[:, 0] == parent]
        
        n_values = len(current_values.index)
        n_colors = min(max(n_values, 3), 9)
        
        current_colors_map = cl.scales[str(n_colors)]['seq'][color]
        if n_values < 3:
            current_colors_map = current_colors_map[-n_values:]
        if n_values > 9: 
            current_colors_map = cl.interp(current_colors_map, n_values)
        
        level_name = values.columns[1]
        current_colors_map = dict(zip(current_values.sort_values(by='value')[level_name], current_colors_map))
        colors_map.update(dict([(index, current_colors_map[index]) for index in current_values[level_name]]))
        
    return colors_map

def generate_values_for_visualization(articles, levels_name, labels_maps):
    
    #initialisation
    values = [articles.id.count()]
    labels = ['Темы']
    parents = ['']
    colors = ['rgb(255,255,255)']
    
    #level 0
    current_values = articles.groupby([levels_name[0]]).id.count()
    
    values += current_values.values.tolist()
    labels += [labels_maps[0][index] for index in current_values.index]
    parents += len(current_values)*['Темы']
    colors += cl.scales['9']['qual']['Paired']

    colors_map = ['Blues', 'Blues', 'Greens', 'Greens', 'Reds', 'Reds', 'Oranges', 'Oranges', 'Purples']
    colors_map = dict(zip(labels_maps[0].keys(), colors_map))

    #other levels
    for parent_index, (parent_name, level_name) in enumerate(zip(levels_name[:-1], levels_name[1:])):

        level_index = parent_index + 1

        articles[level_name], level_labels_map = replace_nan_label(articles[parent_name], articles[level_name], labels_maps[parent_index])
        labels_maps[level_index].update(level_labels_map)

        current_values = articles.groupby([parent_name, level_name]).id.count().to_frame('value').reset_index()

        values += current_values.value.tolist()
        parents += [labels_maps[parent_index][index] for index in current_values[parent_name]]
        labels += [labels_maps[level_index][index] for index in current_values[level_name]]

        current_colors = get_colors(colors_map, current_values)
        colors += [current_colors[index] for index in current_values[level_name]]

        colors_map = dict([(label, colors_map[parent]) 
        					for label, parent in current_values[current_values.columns[:2][::-1]].values])
    
    return labels, parents, values, colors
    

def levels_visualization(articles, levels_name, labels_maps):
    
    articles = articles.copy()
    labels_maps = labels_maps.copy()
    
    labels, parents, values, colors = generate_values_for_visualization(articles, levels_name, labels_maps)
    
    trace = go.Sunburst(labels=labels, parents=parents, values=values,
                        maxdepth = 2, branchvalues="total", 
                        textinfo="label", hoverinfo="value",
                        textfont = {"size":16},
                        marker = {"line": {"width": 1}, "colors":colors})

    layout = go.Layout(width=600, height=600, margin = go.layout.Margin(t=0, l=0, r=0, b=0))

    return go.Figure([trace], layout) 