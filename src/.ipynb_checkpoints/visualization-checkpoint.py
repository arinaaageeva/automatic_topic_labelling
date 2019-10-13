import pandas as pd

import colorlover as cl
import plotly.graph_objects as go

def replace_nan_labels(data, postfix=''):
    
    data = data.copy(deep=True)
    parent_name, labels_name = data.columns
    
    data.loc[pd.isnull(data[labels_name]), labels_name] = data[parent_name] + postfix
    
    return data[labels_name]

def get_colors(colors_map, data):
    
    parent_name, labels_name, values_name = data.columns
    
    new_colors_map = {}
    for parent, color in colors_map.items():
        
        current_data = data[data[parent_name] == parent]
        
        n_values = len(current_data.index)
        n_colors = min(max(n_values, 3), 9)
        
        colors = cl.scales[str(n_colors)]['seq'][color]
        if n_values < 3:
            colors = colors[-n_values:]
        if n_values > 9: 
            colors = cl.interp(colors, n_values)
            
        current_data = current_data.sort_values(by=values_name)[labels_name]
        new_colors_map.update(dict(zip(current_data, colors)))
        
    return new_colors_map

def generate_values_for_visualization(data, levels_names):
    
    data = data.copy(deep=True)
    
    #initialisation
    values = [data.id.count()]
    labels = ['Темы']
    parents = ['']
    colors = ['rgb(255,255,255)']
    
    #level 0
    level_0_name = levels_names[0]
    
    current_values = data.groupby([level_0_name]).id.count()
    values += current_values.values.tolist()
    
    current_labels = current_values.index.tolist()
    labels += current_labels
    
    colors += cl.scales['9']['qual']['Paired']
    parents += len(current_values)*['Темы']

    colors_map = ['Blues', 'Blues', 'Greens', 'Greens', 'Reds', 'Reds', 'Oranges', 'Oranges', 'Purples']
    colors_map = dict(zip(current_labels, colors_map))
    
    #other levels
    last_level_name = levels_names[-1]
    for (parent_name, level_name) in zip(levels_names[:-1], levels_names[1:]):
        
        if level_name == last_level_name:
            data = data.dropna()
            if not len(data): break
        else:
            data[level_name] = replace_nan_labels(data[[parent_name, level_name]])
        
        current_values = data.groupby([parent_name, level_name]).id.count()
        current_values = current_values.to_frame('value').reset_index()
        values += current_values.value.tolist()
        
        current_labels = current_values[level_name].tolist()
        labels += current_labels
        
        current_colors = get_colors(colors_map, current_values)
        colors += [current_colors[label] for label in current_labels]
        
        parents += current_values[parent_name].tolist()
        
        colors_map = current_values[parent_name].apply(lambda x: colors_map[x])
        colors_map = dict(zip(current_values[level_name], colors_map))
    
    return labels, parents, values, colors

def levels_visualization(data, levels_name):
    
    labels, parents, values, colors = generate_values_for_visualization(data, levels_name)
    
    trace = go.Sunburst(labels=labels, parents=parents, values=values,
                        maxdepth = 2, branchvalues="total", 
                        textinfo="label", hoverinfo="value",
                        textfont = {"size":16},
                        marker = {"line": {"width": 1}, "colors":colors})

    layout = go.Layout(width=600, height=600, margin = go.layout.Margin(t=0, l=0, r=0, b=0))

    return go.Figure([trace], layout) 