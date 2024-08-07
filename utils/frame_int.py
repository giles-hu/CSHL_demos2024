import cv2 
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import ffmpy
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans

def frame_counter(vid_path):
    '''
    counts framses in vid
    '''
    frame_count= 0
    vid=cv2.VideoCapture(vid_path)
    while vid.isOpened():
        ret, img = vid.read()
        if ret: 
            frame_count +=1
        else:
            break
    vid.release()
    return frame_count

def h2642mp4(vid_path,des_path,rotation = None):
    '''
    vid_path: path to vid
    des_path: folder to write in
    converts h264 to mp4
    perserves total number of frames
    '''
    vid=vid_path 
    current_path = os.getcwd()
    path = os.path.dirname(vid)
    bname = os.path.basename(vid)[:-5]
    os.chdir(path)
    if rotation is not None:
        ff = ffmpy.FFmpeg(inputs={f'{bname}.h264': f' -i {bname}.h264'}, outputs={f'{des_path}{bname}.mp4':f'-metadata:s:v:0 rotate={rotation} -vcodec copy'})
    else:
        ff = ffmpy.FFmpeg(inputs={f'{bname}.h264': f' -i {bname}.h264'}, outputs={f'{des_path}{bname}.mp4':'-vcodec copy'})
    try:
        ff.run()
        #print('mp4 file generated')
    except Exception as e:
        pass
        #print(e)
        #print('please check folder path')
    os.chdir(current_path)
    return


def read_pts(path):
    colnames=['Timestamp']
    df=pd.read_csv(path,names=colnames,header=None,index_col=False)
    df['Timestamp'] = df['Timestamp'] -df.iloc[0]['Timestamp']
    df['diff']=df['Timestamp'].diff()
    df['diff']=df['diff']/1000
    df['fps']=1/df['diff']
    df['timestamp']=df['Timestamp']/1000
    df=df.fillna(0)
    df['sec']=[int(str(i).split('.')[0]) for i in df['timestamp']]
    return df

def read_events_trial(path):
    df=read_pts(path+'timestamp.txt')
    df_events1=pd.read_csv(path+'Events_1.csv')
    df_events1.at[0,'Timestamp']=df_events1['Timestamp'][0]/10**9
    df_events2=pd.read_csv(path+'Events_2.csv')
    df['Timestamp']=df['Timestamp']/1000+df_events1['Timestamp'][0]
    idx=np.searchsorted(df['Timestamp'],df_events2['Timestamp'],side='left').tolist()
    df['Events']=np.nan
    df['Events']=df['Events'].astype('object')
    df['frame']=[f for f in range(len(df))]
    for i in range(len(idx)):
        df.at[idx[i],'Events']= df_events2.iloc[i]['Event']
    return df


def read_events_cylinder(vid_path):
    df=read_pts(vid_path+'timestamp.txt')
    df_events1=pd.read_csv(vid_path+'/Events_1.csv')
    df_events1.at[0,'Timestamp']=df_events1['Timestamp'][0]/10**9
    df['Timestamp']=df['Timestamp']/1000+df_events1['Timestamp'][0]
    idx=np.searchsorted(df['Timestamp'],df_events1['Timestamp'],side='left').tolist()
    df['Events']=np.nan
    df['Events']=df['Events'].astype('object')
    df['frame']=[f for f in range(len(df))]
    for i in range(len(idx)):
        if idx[i] >=len(df):    
            df.at[len(df)-1,'Events']= df_events1.iloc[i]['Event']
        else:
            df.at[idx[i],'Events']= df_events1.iloc[i]['Event']
    return df

def get_se(df_path):
    df=pd.read_csv(df_path+'frame_int.csv')
    try:
        start_frame = df[df['Events']=='Sync On']['frame'].iloc[0]
        end_frame = df[df['Events']=='Sync Off']['frame'].iloc[0]
    except Exception as e:
        start_frame = df[df['Events']=='Sync On']['frame'].iloc[0]
        end_frame = df[df['Events']=='Sync Off']['frame'].iloc[0]
    return start_frame, end_frame

def extract_frames_array(img_array,f2p,ratio = 30*1.0/1200, batch=100,max_it=30):
    Index = np.arange(0, img_array.shape[0], 1)
    pbar=tqdm(total=img_array.shape[0],leave=True, position=0)
    for i in range(img_array.shape[0]):
        image = img_as_ubyte(
            cv2.resize(img_array[i,:,:,:],None,fx=ratio,
                      fy=ratio,interpolation=cv2.INTER_NEAREST))
        if i ==0:
            DATA = np.empty(
                (img_array.shape[0], np.shape(image)[0], np.shape(image)[1] * 3)
            )
        DATA[i, :, :] = np.hstack(
            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
        )
        pbar.update(1)
    data = DATA - DATA.mean(axis=0)
    data = data.reshape(img_array.shape[0], -1)  # stacking
    kmeans = MiniBatchKMeans(
        n_clusters=f2p, tol=1e-3, batch_size=batch, max_iter=max_it
    )
    kmeans.fit(data)
    frames2pick = []
    for clusterid in range(f2p):  # pick one frame per cluster
        clusterids = np.where(clusterid == kmeans.labels_)[0]

        numimagesofcluster = len(clusterids)
        if numimagesofcluster > 0:
            frames2pick.append(
                Index[clusterids[np.random.randint(numimagesofcluster)]]
            )
    return frames2pick

def adjust_frames(df, fps, adjust_frame_numbers=True):
    """
    Adjusts frames in the provided CSV file to account for frame drops by inserting NaN rows.
    
    Parameters:
    - file_path (str): The path to the input CSV file.
    - output_path (str): The path to save the adjusted CSV file.
    - fps (int): The frames per second of the recording. Default is 30.
    - adjust_frame_numbers (bool): Whether to adjust the frame numbers if no frame drops are detected. Default is True.
    """

    # Calculate the expected time interval based on the provided fps
    expected_time_interval = 1 / fps

    # Detect frame drops
    threshold = 1.5 * expected_time_interval
    df['frame_drop'] = df['diff'] > threshold

    if not df['frame_drop'].any(): #and not adjust_frame_numbers:
        # No frame drops detected and adjustment not required
        print('No frame drops detected. No adjustments made.')
        
        return df

    # Initialize a list to hold the new rows
    new_rows = []

    # Track the current frame number
    current_frame = 0

    # Iterate through the dataframe to insert NaN rows where frame drops occurred
    for index, row in df.iterrows():
        if row['frame_drop']:
            # Calculate the number of missing frames
            num_missing_frames = int(row['diff'] / expected_time_interval) - 1
            for i in range(num_missing_frames):
                new_rows.append(pd.Series({'frame': current_frame, 'timestamp': np.nan, 'diff': np.nan, 'fps': np.nan, 'sec': np.nan, 'Events': np.nan}))
                current_frame += 1
        new_rows.append(row)
        current_frame += 1

    # Convert the list of new rows into a dataframe
    new_df = pd.DataFrame(new_rows)

    # Drop the 'frame_drop' column
    #new_df = new_df.drop(columns=['frame_drop'])

    # Adjust frame numbers if required
    #if adjust_frame_numbers:
    new_df['frame'] = [i for i in range(len(new_df))]

    print('Adjusted frame data')
    return new_df
