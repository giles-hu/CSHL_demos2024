import warnings
warnings.filterwarnings('ignore')
from deeplabcut.utils.auxiliaryfunctions import read_config
import cv2
import numpy as np
import pandas as pd
def create_empty_h5_file(config,video_name, image_names,data = None):
    #load dlc config
    cfg = read_config(config)
    
    scorer = cfg['scorer']
    ma     = cfg['multianimalproject']
    if ma:
        individuals = cfg['individuals']
        indiv = individuals[ind]
        bodyparts = cfg['multianimalbodyparts']
    else:
        bodyparts=cfg['bodyparts']
    coords = ['x', 'y']
    header = pd.MultiIndex.from_product([[scorer],
                                         bodyparts,
                                         coords],
                                        names=['scorer', 'bodyparts', 'coords'])
    arrays2 = [['labeled-data']  * len(image_names),
               [video_name] * len(image_names),
               image_names]
    index = pd.MultiIndex.from_arrays(arrays2)
    if data is not None:
        df = pd.DataFrame(data, columns=header, index=index)    
    else:
        df = pd.DataFrame(np.nan, columns=header, index=index)    
    return df, scorer
import copy
def convert0tonan(points2d):
    data = copy.deepcopy(points2d)
    if len(points2d.shape) == 3:
        for i,p in enumerate(points2d):
            for j,q in enumerate(p):
                if (q == 0).all():
                    data[i,j,:] = np.nan
    else:
        for i,p in enumerate(points2d):
            if (p == 0).all():
                data[i,:] = np.nan        
    return data

def guarantee_multiindex_rows(df):
    # Make paths platform-agnostic if they are not already
    if not isinstance(df.index, pd.MultiIndex):  # Backwards compatibility
        path = df.index[0]
        try:
            sep = "/" if "/" in path else "\\"
            splits = tuple(df.index.str.split(sep))
            df.index = pd.MultiIndex.from_tuples(splits)
        except TypeError:  #  Ignore numerical index of frame indices
            pass
    # Ensure folder names are strings
    try:
        df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
    except AttributeError:
        pass
    

