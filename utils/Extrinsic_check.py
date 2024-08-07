import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from .Intrinsic_check import iter_video,get_image_idx
import glob
import copy
import cv2
from .FormatAsDLC import create_empty_h5_file,convert0tonan,guarantee_multiindex_rows



class XYZ_object():
    def __init__(self, unit = 'mm',scaling = 1):
        super(XYZ_object, self).__init__()
        if unit.lower() == 'm':
            scaling = scaling/100
        elif unit.lower() == 'cm':
            scaling = scaling/10
        self.__X1 = scaling * np.array([-60  ,-7.5,2.5])
        self.__X2 = scaling * np.array([-57.5,20  ,2.5])
        self.__X3 = scaling * np.array([-25  ,-10 ,2.5])
        self.__X4 = scaling * np.array([-27.5,20  ,2.5])
        self.__Y1 = scaling * np.array([-2.5 ,25  ,45])
        self.__Y2 = scaling * np.array([-2.5 ,25  ,30])
        self.__Y3 = scaling * np.array([-5   ,-10 ,37.5])
        self.__Y4 = scaling * np.array([10   ,5   ,37.5])
        self.__Z1 = scaling * np.array([-25  ,57.5,2.5])
        self.__Z2 = scaling * np.array([2.5  ,65  ,2.5])
        self.__Z3 = scaling * np.array([10   ,57.5,2.5])
        self.__Z4 = scaling * np.array([-25  ,37.5,2.5])
        self.__Z5 = scaling * np.array([10   ,37.5,2.5])
        self.__O1 = scaling * np.array([0    ,-10 ,0])
        self.__O2 = scaling * np.array([10   ,0   ,0])  
        self.kp = np.vstack((self.__X1,self.__X2,self.__X3,self.__X4,
                             self.__Y1,self.__Y2,self.__Y3,self.__Y4,
                             self.__Z1,self.__Z2,self.__Z3,self.__Z4,self.__Z5,
                             self.__O1,self.__O2))
        self.len_kp = self.kp.shape[0]
        self.bodyparts = ['X1', 'X2', 'X3', 'X4', 'Y1', 'Y2', 'Y3', 'Y4', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'O1', 'O2']

    def __call__(self, rot_M = np.eye(3), trans = np.zeros(3)):
        kp_hom =  np.ones((self.len_kp,4))
        kp_hom[:,:3] = self.kp # to Homogeneous coordinates
        M = np.eye(4)
        M[:3,:3] = rot_M 
        M[:3,-1] = trans
        out_kp = np.matmul(M, kp_hom.T).T[:,:3]
        return out_kp
        
        
def threshold_confidence(confidence, threshold = .8):
    confident_idx = []
    n_batch = confidence.shape[0]
    for i in range(n_batch):
        passed_idx = np.where(confidence[i] > threshold)[0]
        confident_idx.append(passed_idx)
    return confident_idx

def initialize_intr(xyz,prediction2d,confident_idx,video_obj, error_threshold = 3):    
    imageSize = video_obj.size_img
    N_frame   = prediction2d.shape[0]
    i = 0
    instrinsicM = None
    dist = np.zeros(5)
    min_median_error = 1e3
    loop_indicator  = True
    instrinsicM_out = None
    while loop_indicator and i < N_frame:
        if len(confident_idx[i]) > 6:
            instrinsicM = cv2.initCameraMatrix2D([xyz()[confident_idx[i]][:, np.newaxis, :].astype(np.float32)], 
                                                 [prediction2d[i][confident_idx[i]][:, np.newaxis, :].astype(np.float32)], 
                                                 imageSize)
            
    
            dist        = np.zeros(5)
            ret,rvecs, tvecs, inliers  = cv2.solvePnPRansac(xyz()[confident_idx[i]], prediction2d[i][confident_idx[i]],instrinsicM,dist)
            imgpoints2, _ = cv2.projectPoints(xyz(), rvecs, tvecs, instrinsicM,dist)
            imgpoints2    = imgpoints2[:,0,:]
            median_error = np.median(np.linalg.norm(prediction2d[i][confident_idx[i]] - imgpoints2[confident_idx[i]], axis = 1))
            median_error = min(10*error_threshold,median_error)
            if median_error < min_median_error or i == 0:
                instrinsicM_out = instrinsicM
            if median_error < error_threshold:
                loop_indicator = False
        i += 1        
    if instrinsicM_out is None:
        assert False, 'Please increase the error threshold or decrease confidence threshold'
    else:
        return instrinsicM_out, dist        
        
def crop_keypoints(keypoints, image_shape):
    original_h, original_w = image_shape[:2]
    mask = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < original_w) & \
           (keypoints[:, 1] >= 0) & (keypoints[:, 1] < original_h)
    keypoints[~mask] = [0, 0]    
        
    return keypoints


def compute_centroid(points):
    return np.mean(points, axis=0)

def kabsch_algorithm(A, B):
    centroid_A = compute_centroid(A)
    centroid_B = compute_centroid(B)
    
    A_centered = A - centroid_A
    B_centered = B - centroid_B
    
    H = np.dot(A_centered.T, B_centered)
    
    U, _, Vt = np.linalg.svd(H)
    R_opt = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R_opt) < 0:
        Vt[-1, :] *= -1
        R_opt = np.dot(Vt.T, U.T)
    
    t_opt = centroid_B - np.dot(R_opt, centroid_A)
    
    return R_opt, t_opt

def calculate_3d_error(A, B):
    R_opt, t_opt = kabsch_algorithm(A, B)
    A_transformed = np.dot(A, R_opt.T) + t_opt
    error = np.linalg.norm(A_transformed - B, axis=1)
    return error

def create_ba_dataset(video_folder,video_name_list): 
    for i,video_file_name in enumerate(video_name_list):
        output_dir = os.path.join(video_folder,video_file_name+'_xyz')
        point2d_each = np.load(os.path.join(output_dir,'img_points.npy'),allow_pickle=True)
        img_idx_raw      = get_image_idx(os.path.join(output_dir,'raw'))
        img_idx_overlay  = get_image_idx(os.path.join(output_dir,'overlay'))
        if i ==0:
            imgpoints = np.zeros((len(video_name_list),len( img_idx_raw), 15,2))
            scores    = np.zeros((len(video_name_list),len( img_idx_raw), 15))
        for k,j in enumerate(img_idx_raw ):
            imgpoints[i,k,:,:] = point2d_each[k,:,:2]
            if j in img_idx_overlay:
                scores[i,k,:]    = 0.95
            else:
                scores[i,k,:]    = 0.
    return imgpoints,scores
    

def Generate_refined_training_data(prediction_path,video_path, 
                                   dlc_project_path,
                                   xyz = None, obj_scale = 1, obj_unit = 'mm', 
                                   instrinsicM = None,dist = None,
                                   min_num_kp_fitting = 6,
                                   error_threshold = 5,
                                   skip_frames = None,
                                   conf_thres = .5,):
    video_obj       = iter_video(video_path)
    video_name      = video_path.split('.')[0].split('/')[-1]
    if xyz is None:
        xyz         = XYZ_object(scaling = obj_scale, unit = obj_unit)
    if skip_frames is None:
        skip_frames = video_obj.fps
        
    prediction_path = os.path.join(prediction_path)
    df1             = pd.read_hdf(prediction_path)
    prediction2d    = df1.values.reshape(-1,15,3)[:,:,:2]
    confidence      = df1.values.reshape(-1,15,3)[:,:,-1]
    confident_idx   = threshold_confidence(confidence,  threshold = conf_thres)
    if instrinsicM is None:
        instrinsicM, dist_init = initialize_intr(xyz,prediction2d,confident_idx,video_obj)
    if  dist is None:
        dist                   = dist_init
          
    finetune_training_data_path = os.path.join(os.path.join(dlc_project_path, 'labeled-data'), video_name)
    if not os.path.exists(finetune_training_data_path):
        os.makedirs(finetune_training_data_path)
    
    config   = os.path.join(dlc_project_path,'config.yaml')    
    
    image_names = []
    N_frame = prediction2d.shape[0]
    select_idx = []
    refined_prediction = np.zeros((N_frame,xyz().shape[0],2))
    last_frame_idx = -100
    for i in range(N_frame):
        frame = video_obj()
        if len(confident_idx[i]) > min_num_kp_fitting:
            ret,rvecs, tvecs, inliers  = cv2.solvePnPRansac(xyz()[confident_idx[i]],
                                                            prediction2d[i][confident_idx[i]],
                                                            instrinsicM,dist)
            imgpoints2, _ = cv2.projectPoints(xyz(), rvecs, tvecs, instrinsicM,dist)
            imgpoints2    = imgpoints2[:,0,:]
            #reproj_error = np.linalg.norm(prediction2d[i][confident_idx[i]] 
            #                              - imgpoints2[confident_idx[i]], axis = 1)
            reproj_error = np.linalg.norm(prediction2d[i] 
                                          - imgpoints2, axis = 1)
            median_error = min(10*error_threshold, np.median(reproj_error))
            if (median_error < error_threshold) and (i - last_frame_idx > skip_frames): 
 
                crop_kp = crop_keypoints(imgpoints2, frame.shape)               
                refined_prediction[i] = convert0tonan(crop_kp)
                current_img_name      = 'img%05d.png'% (i)
                last_frame_idx        = np.array(i)
                select_idx.append(i)
                image_names.append(current_img_name )
                cv2.imwrite(os.path.join(finetune_training_data_path,current_img_name ), frame)
                
    if len(select_idx) > 0:
        select_idx = np.array(select_idx)
        data       = refined_prediction[select_idx]   
        df,scorer  = create_empty_h5_file(config,video_name, image_names , data.reshape(data.shape[0],-1))
        df.astype('float64')
        df.sort_index(inplace=True)
        guarantee_multiindex_rows(df)
        df.to_csv(os.path.join(finetune_training_data_path, ('CollectedData_' + scorer + '.csv')))
        df.to_hdf(os.path.join(finetune_training_data_path, ('CollectedData_' + scorer + '.h5')),
                  key="df_with_missing", mode="w") 
    else:
        print('Failed: do not have enough frames meet criteria')
        os.rmdir(finetune_training_data_path)
        


def topk_confidence(confidence,threshold = 0.8):
    confident_idx = []
    for i in range(confidence.shape[0]):
        idx6 = confidence[i].argsort()[6]
        conf_threshold = min(confidence[i,idx6],threshold)
        passed_idx = np.where(confidence[i] >= conf_threshold)[0]
        confident_idx.append(passed_idx)
    return confident_idx  

def refine_dlc_prediction(video_path, prediction_path,
                          confident_threshold = 0.6,
                          xyz_scaling = 1, 
                          instrinsicM = None, dist = None,
                          error_threshold   = 3,
                          min_number_kp_opt = 6, 
                          output_appendix   = 'refined',
                         ):
    
    video_obj = iter_video(video_path)
    xyz = XYZ_object(scaling = xyz_scaling)
    bodyparts = xyz.bodyparts
    # read dlc prediction
    df1 = pd.read_hdf(prediction_path)
    df_out = copy.deepcopy(df1)
    scorer = df1.columns[0][0]
    prediction2d = df1.values.reshape(-1,len(bodyparts),3)[:,:,:2]
    confidence   = df1.values.reshape(-1,len(bodyparts),3)[:,:,-1]
    if confident_threshold is None:
        confident_idx = topk_confidence(confidence)
    else:
        confident_idx = threshold_confidence(confidence,  threshold = confident_threshold)
    if instrinsicM is None:
        instrinsicM, dist_init = initialize_intr(xyz,prediction2d,confident_idx,video_obj) 
        if dist is None:
            dist = dist_init
    N_frame = prediction2d.shape[0]
    refined_prediction = np.zeros((N_frame,15,2))
    for i in range(N_frame):
        if len(confident_idx[i]) >= min_number_kp_opt:
            ret,rvecs, tvecs, inliers  = cv2.solvePnPRansac(xyz()[confident_idx[i]], prediction2d[i][confident_idx[i]],instrinsicM,dist)
            imgpoints2, _ = cv2.projectPoints(xyz(), rvecs, tvecs, instrinsicM,dist)
            imgpoints2    = imgpoints2[:,0,:]
            reproj_error  = np.linalg.norm(prediction2d[i] - imgpoints2, axis = 1)
            median_error  = min(10*error_threshold, np.median(reproj_error))
            if (median_error < error_threshold): 
                for j, bp in enumerate(bodyparts):
                    df_out.at[i, (scorer,bp, 'x')] = imgpoints2[j,0]
                    df_out.at[i, (scorer,bp, 'y')] = imgpoints2[j,1]
                    estimated_confidence = max(np.exp(-reproj_error[j]/(error_threshold)),0)
                    df_out.at[i, (scorer,bp, 'likelihood')] = max(df1.at[i, (scorer,bp, 'likelihood')],0.9)
            else:
                for j, bp in enumerate(bodyparts):
                    estimated_confidence = max(np.exp(-reproj_error[j]/(error_threshold)),0)
                    if estimated_confidence >= 0.4:
                        df_out.at[i, (scorer,bp, 'x')] = imgpoints2[j,0]
                        df_out.at[i, (scorer,bp, 'y')] = imgpoints2[j,1]                    
                    df_out.at[i, (scorer,bp, 'likelihood')] = min(df1.at[i, (scorer,bp, 'likelihood')],estimated_confidence)                
    
    df_out.astype('float64')
    df_out.sort_index(inplace=True)
    guarantee_multiindex_rows(df_out)    
    output_path = prediction_path.split('.')[0]+'_'+output_appendix+'.h5'
    df_out.to_hdf(output_path, key="df_with_missing", mode="w") 
     
        
        
  
    
