import pandas as pd

from sklearn.preprocessing import LabelEncoder

def encode_labels(level):
    
    level = level.copy()

    flags = ~pd.isnull(level)
    
    label_encoder = LabelEncoder().fit(level[flags])
    level[flags] = label_encoder.transform(level[flags])
    
    labels = label_encoder.classes_
    labels = pd.DataFrame({'id':range(len(labels)), 'label':labels})
    
    return level, labels

def replace_nan_label(parents, levels, parent_labels_map, postfix=' '):
    
    parents = parents.copy()
    levels = levels.copy()

    flags = pd.isnull(levels)

    levels[flags] = max(0, levels.max() + 1) + parents[flags]
    levels = levels.astype(int)
    
    level_labels_map = {}
    level_parent_map = dict(pd.concat([levels, parents], axis=1)[flags].drop_duplicates().values)
    for level, parent in level_parent_map.items():
        level_labels_map[level] = parent_labels_map[parent] + postfix

    return levels, level_labels_map