import ffmpy
import os
import pandas as pd
import cv2 
import numpy as np
import os
from decord import VideoReader
from decord import cpu, gpu
from tqdm import tqdm 
import matplotlib.pylab as plt
import tifffile as tf
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
#import frame_int as fi
import matplotlib.pylab as plt
import glob
def get_se(df_path):
    if os.path.isfile(os.path.join(os.path.join(df_path,'frame_int.csv'))):
        df=pd.read_csv(os.path.join(df_path,'frame_int.csv'), encoding='utf8')
        start_frame = df[df['Events']=='Sync On']['frame'].iloc[0]
        end_frame = df[df['Events']=='Sync Off']['frame'].iloc[0]
        return np.array([start_frame, end_frame])
    else:
        print('frame_int.csv does not exist in',df_path)
        return np.array([np.nan,np.nan])
def extract_frames(vid_path,frame_list):
    vr = VideoReader(vid_path, ctx=cpu(0))
    frames = vr.get_batch(frame_list).asnumpy()
    return frames    
def get_se2(df):
    start_frame = df[df['Events']=='Sync On']['frame'].iloc[0]
    end_frame = df[df['Events']=='Sync Off']['frame'].iloc[0]
    return np.array([start_frame, end_frame])
def extract_frames_array(img_array,ratio, f2p,batch,max_it):
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
def adjust_frames(df, fps=56, adjust_frame_numbers=True):
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
                new_rows.append(pd.Series({'frame': np.nan, 'timestamp': np.nan, 'diff': np.nan, 'fps': np.nan, 'sec': np.nan, 'Events': np.nan}))
                current_frame += 1
        new_rows.append(row)
        current_frame += 1

    # Convert the list of new rows into a dataframe
    new_df = pd.DataFrame(new_rows)

    # Drop the 'frame_drop' column
    #new_df = new_df.drop(columns=['frame_drop'])

    # Adjust frame numbers if required
    #if adjust_frame_numbers:
    #new_df['frame'] = [i for i in range(len(new_df))]

    print(f'Adjusted frame data')
    return new_df

def h2642mp4(vid_path,rotation:int = None):
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
        ff = ffmpy.FFmpeg(inputs={f'{bname}.h264': f' -i {bname}.h264'}, outputs={f'{bname}.mp4':f'-metadata:s:v:0 rotate={rotation} -vcodec copy'})
    else:
        ff = ffmpy.FFmpeg(inputs={f'{bname}.h264': f' -i {bname}.h264'}, outputs={f'{bname}.mp4':'-vcodec copy'})
    try:
        ff.run()
        #print('mp4 file generated')
    except Exception as e:
        pass
        #print(e)
        #print('please check folder path')
    os.chdir(current_path)
def read_events_cylinder(vid_path):
    if (os.path.isfile(os.path.join(vid_path,'timestamp.txt')) 
        and
        os.path.isfile(os.path.join(vid_path,'Events_1.csv'))):
        df=read_pts(os.path.join(vid_path,'timestamp.txt'))
        df_events1=pd.read_csv(os.path.join(vid_path,'Events_1.csv'))
        
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
    else:
        print('timestamp.txt or Events_1.csv does not exist in ',vid_path)
        return None

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

def process_cam_data(src_path):
    file_path = os.path.join(src_path, 'frame_int.csv')
    
    df_d = pd.read_csv(file_path, index_col=0)
    fps = df_d['fps'].mean().astype('int')
    #print(fps)
    df_dn = adjust_frames(df_d, fps=fps, adjust_frame_numbers=True)
    camdf1, camdf2 = get_se2(df_dn)
    return df_dn, camdf1, camdf2

def find_nan_timestamps(df):
    nan_indices = []
    for i, t in enumerate(df['timestamp']):
        if math.isnan(t):
            nan_indices.append(i)
    return nan_indices
#len(processed_data['checkerboard']['camD']['df']['frame']) #[727]


def assemble_img2video(frames_array,
                       original_video_path,
                       output_video_path):
    # Step 1: Get the parameters of the original video
    cap    = cv2.VideoCapture(original_video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = frames_array.shape[2] #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = frames_array.shape[1] #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec  = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    cap.release()

    # Step 2: Create a VideoWriter object with the same parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Assuming the original codec is mp4v

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Step 3: Write the frames to the video file
    for frame in frames_array:
        out.write(frame)
    
    out.release()

    print(f'Video saved as {output_video_path}')
# Open the video file
def save_each_frame(video_path,output_dir, df_each):
    cap = cv2.VideoCapture(video_path)
    
    fps    = cap.get(cv2.CAP_PROP_FPS)

    codec  = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    os.makedirs(output_dir, exist_ok=True)
    
    start_index = df_each.loc[df_each['Events'] == 'Sync On'].index[0]
    end_index   = df_each.loc[df_each['Events'] == 'Sync Off'].index[0]
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
        
    i = 0
    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        
        if i ==0:
            width  = frame.shape[1] #int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = frame.shape[0] #int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_dir, fourcc, fps, (width, height))
    
        # If 'ret' is False, it means we have reached the end of the video
        if not ret:
            break
        else:
            i = i+1
        
        if i in np.arange(start_index, end_index):
            current_frame = df_each['frame'][i] - df_each['frame'][start_index]
            frame_filename = os.path.join(output_dir, f'frame_{current_frame:06d}.png')
            cv2.imwrite(frame_filename, frame)
        

def pair_frame_index(df_each):
    start_index = df_each.loc[df_each['Events'] == 'Sync On'].index[0]
    end_index   = df_each.loc[df_each['Events'] == 'Sync Off'].index[0]
    start_frame = df_each['frame'][start_index]
    end_frame   = df_each['frame'][end_index  ]
    total_frame = end_frame - start_frame 
    frame_pair = np.zeros((total_frame,2))
    frame_pair[:,0] = np.arange(start_frame, end_frame)
    
    for i in range(total_frame):
        current_idx = df_each.loc[df_each['frame'] == frame_pair[i,0]].index[0]
        if type(current_idx) is str:
            frame_pair[i,1] = np.nan
        else:
            frame_pair[i,1] = current_idx    
    return frame_pair
    
import math
def synchronize_videos(root_path, project_name,cam_name):
    processed_data = {}
    frame_range= {}
    frame_pair = {}
    all_nan_indices = {} 
    
    for p in project_name:
        processed_data[p] = {}
        frame_range[p] = {}
        frame_pair[p] = {}
        all_nan_indices[p] = set()
        for c in cam_name:
            processed_data[p][c] = {}
            src_path = os.path.join(root_path,os.path.join(p,c))
            if c.lower() != 'cama':
                vid_path = os.path.join(src_path,'test.h264')
                try:
                    h2642mp4(vid_path,rotation = 90)
                except:
                    pass
            else:
                vid_path = os.path.join(src_path,'test.h264')
                try:
                    h2642mp4(vid_path)
                except:
                    pass
            df = read_events_cylinder(src_path)
            if df is not None:
                df.to_csv(os.path.join(src_path, f'frame_int.csv'))            
            frame_range[p][c] = get_se(src_path)
            df_n, camdf1, camdf2          = process_cam_data(src_path)
            processed_data[p][c][f'df']    = df_n
            processed_data[p][c][f'camf1'] = camdf1
            processed_data[p][c][f'camf2'] = camdf2   
            
            nan_indices = find_nan_timestamps(df_n)
            all_nan_indices[p].update(nan_indices)
            
            frame_pair[p][c] = pair_frame_index(df_n)
            
    droped_frame = {} 
    kept_frame = {}
    for p in project_name:
        droped_frame[p] = [] 
        for c in cam_name:
            src_path = os.path.join(root_path,os.path.join(p,c))
            if os.path.isfile(os.path.join(os.path.join(src_path,'frame_int.csv'))): 
                droped_frame[p] = np.append(droped_frame[p],np.where(np.isnan(frame_pair[p][c][:,0]))[0] )
        all_array = np.arange(frame_pair[p][c].shape[0])
        kept_frame[p] = np.setdiff1d(all_array,droped_frame)  
        
    frame_ind = {}
    for p in project_name:
        frame_ind[p] = {}
        for c in cam_name:
            src_path = os.path.join(root_path,os.path.join(p,c))
            if os.path.isfile(os.path.join(os.path.join(src_path,'frame_int.csv'))): 
                frame_ind[p][c] = np.arange(processed_data[p][c][f'camf1'],
                                            processed_data[p][c][f'camf2'],
                                            1)
                total_length = len(frame_ind[p][c])
                selected_indices = np.arange(0, total_length ,1)
                selected_indices = [i for i in selected_indices if i not in all_nan_indices[p]]
                picked_frames = extract_frames(os.path.join(src_path,'test.mp4'),
                                               frame_ind[p][c][selected_indices])
                
                picked_frames = picked_frames.astype(np.uint8)
                assemble_img2video(picked_frames,os.path.join(src_path,'test.mp4'),
                                   os.path.join(os.path.join(root_path,p),f'{p}-{c}.mp4'))
