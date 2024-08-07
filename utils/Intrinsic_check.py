import warnings
warnings.filterwarnings('ignore')
import numpy as np
import time
import os 
import cv2 
import glob
import matplotlib.pyplot as plt

import copy

from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans

# write a class that is a video， for each time call it will return a undistorted image until there is not image
class iter_video(): 
    def __init__(self,video_path, cam_config = None):
        
        self.currentframe = -1
        self.video        = cv2.VideoCapture(video_path)
        self.ret          = True
        self.cam_config   = cam_config
        self.fps          = self.video.get(cv2.CAP_PROP_FPS)
        self.width        = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        self.height       = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        self.size_img     = (int(self.width),int(self.height))
        #self.frame_count  = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
    def __call__(self):
        ret,frame = self.video.read()
        self.ret  = ret
        if ret:
            self.currentframe += 1
            image = self.undistort_img(frame) #cv2.cvtColor( , cv2.COLOR_BGR2GRAY)
            return image
        else:
            self.video.release()
            return None
    
    def undistort_img(self, img_distorted):
        # Load the camera calibration parameters
        if self.cam_config is None:
            img_undistorted = img_distorted
        else:
            camera_matrix = self.cam_config['intr']
            dist_coeff    = self.cam_config['distort']
            # Undistort the image
            img_undistorted = cv2.undistort(img_distorted, camera_matrix, dist_coeff)
        return img_undistorted  
        
class CheckerBoard():
    def __init__(self,Rows,Columns,Checker_Width,criteria = None):
        # define checker board
        self.checkersize = np.array([Rows-1,Columns-1]).astype(int)
        self.Checker_Width = Checker_Width
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((int((Rows-1)*(Columns-1)) ,3), np.float32)
        self.objp[:,:2] = (np.mgrid[0:(Rows-1) ,0:(Columns-1)].T.reshape(-1,2) 
                           * self.Checker_Width)
        # define criteria
        if criteria is None:
            self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,0.001)
        else:
            self.criteria = criteria
    def __call__(self):
        pass
        
def corner_detector(image,CheckerBoard):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret_c, corners = cv2.findChessboardCorners(gray, 
                                               (CheckerBoard.checkersize[0],
                                                CheckerBoard.checkersize[1]),None)
    if ret_c == True:
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), CheckerBoard.criteria)
        return ret_c,corners2
    else:
        return ret_c,[]
    
# load images and record the index of it.

import glob
def get_image_idx(output_dir):
    list_names = sorted(glob.glob( os.path.join(output_dir,'*.png')))
    image_idx = []
    for ln in list_names :
        image_idx.append(ln.split('/')[-1].split('_')[-1].split('.')[0])
    return np.array(image_idx).astype(int)
# load images kmean
#num_frame2pick = 40
def load_kmean_imgaes(output_dir, numframes2pick = 40, 
                      resizewidth=32,batchsize=100,max_iter=50,):
    # stolen from deeplabcut
    img_idx = get_image_idx(output_dir)
    img_data = []
    list_names = sorted(glob.glob( os.path.join(output_dir,'*.png')))
    for i,ln in enumerate(list_names):
        img = cv2.cvtColor(cv2.imread(ln), cv2.COLOR_RGB2GRAY)
        if i ==0:
            original_height, original_width = img.shape[:2]
            new_width = resizewidth
            new_height = int((new_width / original_width) * original_height)
        resized_image = img_as_ubyte(cv2.resize(img, (new_width, new_height)))
        flat_img = resized_image.reshape(-1)
        img_data.append(flat_img)
    img_data = np.array(img_data)
    img_data = img_data - img_data.mean(axis=0)
    if batchsize > img_data.shape[0]:
        batchsize = int(img_data.shape[0] / 2)
    kmeans = MiniBatchKMeans(n_clusters=numframes2pick, tol=1e-3, 
                             batch_size=batchsize, max_iter=max_iter)
    kmeans.fit(img_data)    
    frames2pick = []
    for clusterid in range(numframes2pick):  # pick one frame per cluster
        clusterids = np.where(clusterid == kmeans.labels_)[0]
    
        numimagesofcluster = len(clusterids)
        if numimagesofcluster > 0:
            frames2pick.append(
                img_idx[clusterids[np.random.randint(numimagesofcluster)]]
            )        
    return frames2pick
        
def calibrate_kmean_pick_frames(output_dir,checkerB,
                                numframes2pick = 20, 
                                resizewidth=32,batchsize=100,max_iter=50,
                                verbose = False):
    imgpoints = np.load(os.path.join(output_dir ,'img_points.npy'),allow_pickle=True)
    num_train_img = len(sorted(glob.glob( os.path.join(os.path.join(output_dir,'train'),'*.png'))))
    if  numframes2pick > num_train_img:
        print('number to pick is greater than number of training images')
        print('number to pick has been reduced')
        numframes2pick = num_train_img    
    frames2pick = load_kmean_imgaes(os.path.join(output_dir,'train'), numframes2pick = numframes2pick, 
                                    resizewidth=resizewidth,batchsize=batchsize,
                                    max_iter=max_iter,)
    imgpoints_finetune = []
    for i,j in enumerate(imgpoints):
        if i in frames2pick:
            imgpoints_finetune.append(j)
    objpoints_finetune = np.repeat(np.expand_dims(checkerB.objp, axis=0), len(imgpoints_finetune), axis=0)
    # get intrinsic parameters

    img = cv2.imread(sorted(glob.glob( os.path.join(os.path.join(output_dir,'train'),'*.png')))[0])
    img_size = (img.shape[1],img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_finetune,
                                                       np.array(imgpoints_finetune), 
                                                       img_size , None, None)
    if verbose:
        return ret, mtx, dist, rvecs, tvecs, objpoints_finetune, np.array(imgpoints_finetune)
    else:
        return ret, mtx, dist

def calibrate_all_frames(output_dir,checkerB,verbose = False):
    imgpoints = np.load(os.path.join(output_dir ,'img_points.npy'),allow_pickle=True)
    frames2pick = get_image_idx(os.path.join(output_dir,'train'))
    imgpoints_finetune = []
    for i,j in enumerate(imgpoints):
        if i in frames2pick:
            imgpoints_finetune.append(j)
    objpoints_finetune = np.repeat(np.expand_dims(checkerB.objp, axis=0), len(imgpoints_finetune), axis=0)
    #print(np.array(imgpoints_finetune).shape)
    # get intrinsic parameters
    img = cv2.imread(sorted(glob.glob( os.path.join(os.path.join(output_dir,'train'),'*.png')))[0])
    img_size = (img.shape[1],img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_finetune,
                                                       np.array(imgpoints_finetune), 
                                                       img_size , None, None)
    if verbose:
        return ret, mtx, dist, rvecs, tvecs, objpoints_finetune, np.array(imgpoints_finetune)
    else:
        return ret, mtx, dist

        
def validate_calibrate(output_dir,checkerB,instrinsicM,dist ):
    imgpoints_val = []
    validation_dir = os.path.join(output_dir,'val')
    for img_name in sorted(glob.glob( os.path.join(validation_dir,'*.png'))):
        img = cv2.imread(img_name)
        size_img = img.shape[::-1]
    
        ret_c, corners = corner_detector(img,checkerB)
        if ret_c:
            imgpoints_val.append(corners)
    # Estimate rvecs and tvecs for validation
    rvecs_val, tvecs_val = [], []
    objpoints_val = np.repeat(np.expand_dims(checkerB.objp, axis=0), len(imgpoints_val), axis=0)
    
    for i in range(len(objpoints_val)):       
        _, rvec, tvec = cv2.solvePnP(objpoints_val[i], imgpoints_val[i],instrinsicM,dist)
        rvecs_val.append(rvec)
        tvecs_val.append(tvec)    
    ret = calculate_reprojection_error(objpoints_val, imgpoints_val, rvecs_val, tvecs_val, instrinsicM,dist)
    return ret
def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
        total_error += error ** 2
        total_points += len(imgpoints[i])
    mean_error = np.sqrt(total_error / total_points)
    return mean_error

           
def create_dataset_for_calibration(video_folder, video_file_name,checkerBoard, 
                                   sample_ratio = 0.1):
    # detect checker board from a video, and save the image which is successfuly detected.
    
    output_dir = os.path.join(video_folder,video_file_name.split('.')[0]+'_checker_img')
    video_path = os.path.join(video_folder,video_file_name)    
    # read video from video path
    video_obj = iter_video(video_path)
    if not os.path.isfile(os.path.join(output_dir,'img_points.npy')):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir,'overlay'), exist_ok=True)
        os.makedirs(os.path.join(output_dir,'raw'), exist_ok=True)
        imgpoints = []
        while video_obj.ret:
            img = video_obj()
            if img is not None:
                size_img = img.shape[::-1]
                img_idx = video_obj.currentframe
                if np.random.rand() <= sample_ratio:
                    ret_c, corners = corner_detector(img,checkerBoard)
                else:
                    ret_c, corners = False, []
                imgpoints.append(corners)
                if ret_c:
                    raw_frame_filename = os.path.join(os.path.join(output_dir,'raw'), 
                                                      f'frame_{img_idx:06d}.png')
                    cv2.imwrite(raw_frame_filename, img)
                    cv2.drawChessboardCorners(img, (checkerBoard.checkersize[0],
                                                    checkerBoard.checkersize[1]), corners, ret_c)
                    overlay_frame_filename = os.path.join(os.path.join(output_dir,'overlay'), 
                                                          f'frame_{img_idx:06d}.png')
                    cv2.imwrite(overlay_frame_filename, img)  
        imgpoints = np.array(imgpoints,dtype=object )
        np.save(os.path.join(output_dir,'img_points.npy'),imgpoints  )
        print('successfully finished')

def kmean_pick_dataset(checker_img_folder,numframes2pick = 64):
    img_idx    = get_image_idx(os.path.join(checker_img_folder,'raw'))
    if len(img_idx) > numframes2pick:
        picked_idx = load_kmean_imgaes(os.path.join(checker_img_folder,'raw'),numframes2pick = numframes2pick)
        
        not_picked_idx = np.setdiff1d( img_idx, picked_idx,  )
        for i in not_picked_idx:
            os.remove(os.path.join(os.path.join(checker_img_folder,'raw'),f'frame_{i:06d}.png' ))
            os.remove(os.path.join(os.path.join(checker_img_folder,'overlay'),f'frame_{i:06d}.png' ))
        print('successfully finished')
    else:
        print('Warning: Number of images is less than number to pick')        
import random
def random_split_list(original_list, ratio=0.8):
    # Calculate the number of elements for the first part
    split_size = int(ratio * len(original_list))
    # Randomly sample elements for the first part
    first_part = random.sample(original_list, split_size)
    # Create the second part with the remaining elements
    second_part = [item for item in original_list if item not in first_part]
    return first_part, second_part
        
import shutil        
def split_train_test(checker_img_folder, train_test_ratio = 0.8): 
    if  (os.path.exists(os.path.join(checker_img_folder,'overlay'))
         and os.path.exists(os.path.join(checker_img_folder,'raw')) ):
        validation_dir = os.path.join(checker_img_folder,'val')        
        train_dir      = os.path.join(checker_img_folder,'train') 
        os.makedirs(validation_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        list_names = sorted(glob.glob( os.path.join(os.path.join(checker_img_folder,'overlay'),'*.png')))
        list_names = [x.split('/')[-1] for x in list_names]
        names_train, names_val = random_split_list(list_names, ratio=train_test_ratio)
        for n in names_train:
            src_file  = os.path.join(os.path.join(checker_img_folder,'overlay'), n)
            dest_file = os.path.join(train_dir, n)
            shutil.move(src_file, dest_file)   
        for n in names_val:
            src_file  = os.path.join(os.path.join(checker_img_folder,'raw'), n)
            dest_file = os.path.join(validation_dir, n)
            shutil.move(src_file, dest_file)   
        shutil.rmtree(os.path.join(checker_img_folder,'raw'))
        shutil.rmtree(os.path.join(checker_img_folder,'overlay'))
        print('successfully finished')
    else:
        print('Not perform function *create_dataset_for_calibration* yet or trainining and validation dataset already exist')

                                
                             

