from .Extrinsic_check import create_ba_dataset
from .aniposelib.cameras import CameraGroup
from anipose.anipose import  load_config
from anipose.calibrate import load_2d_data,process_points_for_calibration
from anipose.common import get_calibration_board
import numpy as np
import os
import pickle as pkl
import toml
import copy
def refine_calibration(anipose_path,intrinsic_result_path, all_points, all_scores, file_appendix = '-refine'):    
    path_calibration_raw = os.path.join(anipose_path ,'videos/calibration/calibration.toml' )
    intrinsic_result     = np.load(intrinsic_result_path,allow_pickle=True).item()  
    
    config = load_config(os.path.join(anipose_path,'config.toml') )
    board  = get_calibration_board(config)
    rows_fname = os.path.join(anipose_path,'videos/calibration/detections.pickle')
    assert  os.path.exists(rows_fname)
    with open(rows_fname, 'rb') as f:
        all_rows = pkl.load(f)
    # load original calibration
    with open(path_calibration_raw, 'r') as f:
        calibration_result_raw = toml.load(f)
    cgroup = CameraGroup.load(path_calibration_raw)
    # update intrinsic with newer Intrinsic
    for i in range(len(cgroup.cameras)):
        cam_name = 'cam' + calibration_result_raw[f'cam_{i}']['name']
        cgroup.cameras[i].set_camera_matrix(np.array(intrinsic_result[cam_name]['intr']     ) )
        cgroup.cameras[i].set_distortions(np.array(intrinsic_result[cam_name]['dist'])[0])  
    # Re-estimate Extrinsic with fixed Intrinsic
    error = cgroup.calibrate_rows(all_rows, board,
                                  init_intrinsics=False, init_extrinsics=True,
                                  max_nfev=200, n_iters=6,
                                  n_samp_iter=200, n_samp_full=1000,
                                  verbose=True, only_extrinsics=True,)
    imgp = process_points_for_calibration(all_points, all_scores)
    error = cgroup.bundle_adjust_iter(imgp, ftol=1e-3, n_iters=10,
                                      n_samp_iter=300, n_samp_full=1000,
                                      max_nfev=500,only_extrinsics=True,
                                      verbose=True)
    cgroup.metadata['adjusted'] = True
    cgroup.metadata['error'] = float(error)
    if file_appendix is None:
        calibration_save_path = os.path.join(anipose_path,f'videos/calibration/calibration.toml')
    else:
        calibration_save_path = os.path.join(anipose_path,f'videos/calibration/calibration{file_appendix}.toml')
    cgroup.dump(calibration_save_path)
    print(f'Finished: New calibration result saved to {calibration_save_path }')


