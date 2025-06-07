import argparse
import numpy as np
import random
from pathlib import Path
from scipy.io import savemat
import scipy as sp
from scipy.spatial.transform import Rotation as Rot
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import os
import json

import torch
from pytorch3d.structures import Meshes, Pointclouds, join_meshes_as_batch
from pytorch3d.transforms import so3_relative_angle
from pytorch3d.loss       import chamfer_distance

import tools._init_paths

from configs          import cfg, update_config
# from dataset.build    import build_dataset
from nets             import Model
from nets.utils       import transform_world2primitive, inside_outside_function_dual
from utils.visualize  import plot_3dmesh, plot_3dpoints, plot_occupancy_labels, imshow
from utils.utils      import (
    set_seeds_cudnn,
    initialize_cuda,
    load_camera_intrinsics
)
from nets.modules.layers import PrimitiveParameters

from nets.renderer import *

from nets.losses import AdaptedRotationLoss

import sys

np.random.seed(273)

net = Model(cfg, 1200)

np.set_printoptions(threshold=sys.maxsize)

# plt.rcParams["text.usetex"] = True

def load_dataset_poses(dataset_path):

    # just load the json poses
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # separate out the data, put into arrays

    n = len(data)

    file_paths = []
    all_rot_v2t = np.zeros((n,4))
    all_trans_v2t = np.zeros((n,3))
    for i in range(0, n):
        file_paths.append(data[i]['filename'])
        all_rot_v2t[i,:] = data[i]['q_vbs2tango_true']
        all_trans_v2t[i,:] = data[i]['r_Vo2To_vbs_true']

    return file_paths, all_rot_v2t, all_trans_v2t

def load_dataset_metadata(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # separate out the data, put into arrays

    # print(data["tRelState"])

    n = len(data["sAbsState"]["oe_osc_kep"])

    # print(data["sAbsState"]["oe_osc_kep"])


    s_oe_all = np.zeros((n,6))
    roe_all = np.zeros((n,6))
    w_all = np.zeros((n,3)) # Angular velocity of the servicer's principal axes w.r.t. the target's principal axes expressed in the target's principal axes
    w_abs_all = np.zeros((n,3)) 
    q_eci2pri_all = np.zeros((n,4)) 
    rv_all = np.zeros((n,6))
    for i in range(0, n):
        s_oe_all[i,:] = data["sAbsState"]["oe_osc_kep"][i]
        roe_all[i,:] = data["tRelState"]["roe_osc_qns"][i]
        w_all[i,:] = data["tRelState"]["w_tpri2spri_tpri"][i]
        w_abs_all[i,:] = data["sAbsState"]["w_pri"][i]
        q_eci2pri_all[i,:] = data["sAbsState"]["q_eci2pri"][i]
        rv_all[i,:] = data["sAbsState"]["rv_eci2com_eci"][i]

    return w_all, s_oe_all, roe_all, w_abs_all, q_eci2pri_all, rv_all

def omega_matrix(w):
    # w is a 3x1 vector
    w = w.flatten()
    O = np.zeros((4,4))
    O[1:4, 0] = w
    O[0, 1:4] = -w
    wx = np.array([[0,     -w[2],   w[1]],
                   [ w[2],     0, - w[0]],
                   [-w[1],  w[0],      0]])
    O[1:4, 1:4] = -wx

    return O

def unscented_transform(mu, S, l):

    n = np.size(mu)
    x0 = mu
    w0 = l / (l + n)

    points = [x0.flatten()]
    weights = [w0]

    mat = sp.linalg.cholesky((l + n) * S)

    for i in range(0,n):
        xi = mu.flatten() + mat[:,i]
        xj = mu.flatten() - mat[:,i]
        w = 1 / (2 * (l + n))


        points.extend([xi, xj])
        weights.extend([w, w])

    # print(f"UT points: {points}")
    return points, weights

def inverse_unscented_transform(points, weights):
    # points must be either a list of vectors or a (2N+1)xN array

    if type(points) is list:
        points = np.array(points)

    if type(weights) is list:
        weights = np.array(weights)

    
    n = points[0].size
    n_points =int( points.size / n)
    
    points = points.reshape((n_points, n))
    weights = weights.reshape((1,n_points))

    weights_s = weights.copy()
    # print(f"weights_s: {weights_s}")
    weights_s[0,0] = weights_s[0,0] + 2
    
    # print(n)
    
    # print(points[0].shape)
    # print(np.array(weights).shape)
    mu = (weights @ points).reshape((n,1))
    # print(f"mu: {mu}")

    S = np.zeros((n,n))

    for i in range(0, 2*n+1):
        # print(f"UT^-1 - adding to S: {weights[i] * (points[i].reshape(n,1) - mu.reshape(n,1)) @ (points[i].reshape(n,1) - mu.reshape(n,1)).T}")
        # print(((points[i].reshape(n,1) - mu.reshape(n,1))).shape)
        # print(((points[i].reshape(n,1) - mu.reshape(n,1)).T).shape)
        # print(weights[0][i].shape)
        S = S + weights_s[0][i] * (points[i].reshape(n,1) - mu.reshape(n,1)) @ (points[i].reshape(n,1) - mu.reshape(n,1)).T
        
    return mu, S

def cross_cov(x_points, y_points, weights):
    # points must be either a list of vectors or a (2N+1)xN array

    if type(x_points) is list:
        x_points = np.array(x_points)

    if type(y_points) is list:
        y_points = np.array(y_points)

    if type(weights) is list:
        weights = np.array(weights)

    

    x_points = x_points.reshape((25,12)) # N = 12
    y_points = y_points.reshape((25,7)) # M = 7
    weights = weights.reshape((1,25))

    weights_s = weights.copy()
    # print(f"weights_s: {weights_s}")
    weights_s[0,0] = weights_s[0,0] + 2


    n = x_points[0].size # dimension of state
    m = y_points[0].size # dimension of measurement
    # print(n)
    
    # print(points[0].shape)
    # print(np.array(weights).shape)
    mu_x = (weights @ x_points).reshape((n,1))
    mu_y = (weights @ y_points).reshape((m,1))
    # print(f"mu: {mu}")

    S = np.zeros((n,m))

    for i in range(0, 2*n+1):
        # print(f"UT^-1 - adding to S: {weights[i] * (points[i].reshape(n,1) - mu.reshape(n,1)) @ (points[i].reshape(n,1) - mu.reshape(n,1)).T}")
        # print(((points[i].reshape(n,1) - mu.reshape(n,1))).shape)
        # print(((points[i].reshape(n,1) - mu.reshape(n,1)).T).shape)
        # print(weights[0][i].shape)

        # print((x_points[i].reshape(n,1) - mu_x.reshape(n,1)).shape)
        # print( ( (y_points[i].reshape(m,1) - mu_y.reshape(m,1)).T).shape )
        S = S + weights_s[0][i] * (x_points[i].reshape(n,1) - mu_x.reshape(n,1)) @ (y_points[i].reshape(m,1) - mu_y.reshape(m,1)).T

    return S

def determine_RTN_rot(rv_eci, q_eci2pri):
    r = rv_eci[0:3]
    v = rv_eci[3:6]

    n = np.cross(r,v)

    R = r / np.linalg.norm(r)
    N = n / np.linalg.norm(n)
    T = np.cross(N, R)

    R_eci_rtn = np.hstack((R.reshape((3,1)), T.reshape((3,1)), N.reshape((3,1)))).T

    R_eci_p = Rot.from_quat(q_eci2pri, scalar_first=True).as_matrix()

    R_p_rtn = R_eci_p.T @ R_eci_rtn

    return R_p_rtn
    

def quat_mult(q1, q2):
    # from https://stackoverflow.com/questions/39000758/how-to-multiply-two-quaternions-by-python-or-numpy
    w0, x0, y0, z0 = q1
    w1, x1, y1, z1 = q2
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def predict_state(state, q, roe_stm, J_target, J_servicer, w_s_abs, dt):
    # state:
    # 0:5: roe
    # 6:8: dp (MRP vector)
    # 9:12: w (relative angular velocity)

    # Update ROEs with STM (for simplicity)
    # print(state)
    # state = state[0]
    # print(state)
    # print(state[0:6])
    # print(f"ROE before propagate: {state[0:6]}")
    roe_predict = roe_stm @ state[0:6].reshape((6,1))
    # print(f"ROE after propagate: {roe_predict}")

    # Update error quaternion (?)
    abs_rot_est = Rot.from_quat(q.T, scalar_first=True)
    q_rot_err = Rot.from_mrp(state[6:9].T).as_quat(scalar_first=True)
    q_est = quat_mult( q_rot_err, q)

    # print(f"q_est: {q_est}")

    R_est = Rot.from_quat(q_est, scalar_first=True).as_matrix() # R_t_wrt_s
    omega = omega_matrix(-state[9:12])

    q_dot = .5 * omega @  q_est # q_dot_t_wrt_s
    # print(f"q_dot: {q_dot}")
    # out of laziness, try just increment by dt instead of properly integrating
    # this will probably not work well lol
    # TODO

    # assume absolute angular accleration of servicer is 0
    # print(gt_w_s_abs_all)
    
    
    w_dot_s = np.zeros(3,) # w_dot_s_abs
    # print(w_dot_s[2])
    w_dot_s[0] = ((J_servicer[1,1] - J_servicer[2,2])/J_servicer[0,0])*w_s_abs[1]*w_s_abs[2]
    w_dot_s[1] = ((J_servicer[2,2] - J_servicer[0,0])/J_servicer[1,1])*w_s_abs[2]*w_s_abs[0]
    w_dot_s[2] = ((J_servicer[0,0] - J_servicer[1,1])/J_servicer[2,2])*w_s_abs[0]*w_s_abs[1]
    # print(w_dot_s)
    # Hardcoding to debug: 
    w_dot_s = np.zeros(3,)


    w_t_abs_est = R_est @ w_s_abs - state[9:12].reshape((3,1))
    # print(np.cross(w_t_abs_est.reshape(3,),  (J_target @ w_t_abs_est).reshape(3,) ))
    # print(np.cross(w_t_abs_est, mu_ prior[9:12]))
    # w_dot_s_wrt_t
    w_dot = R_est @ w_dot_s - np.linalg.inv(J_target) @ ( -np.cross(w_t_abs_est.reshape(3,),  (J_target @ w_t_abs_est).reshape(3,) )  ) - np.cross(w_t_abs_est.reshape(3,), state[9:12].reshape(3,))
    # print(f"w_dot: {w_dot}")
    # TEMP - hardcoding
    w_dot = np.zeros((3,1))

    # TODO integrate or increment

    # "integrate" the angular velocity and quaternion
    w_int = state[9:12].reshape((3,1))
    q_int = q
    dt_int = dt / 100
    for j in range(0, 100):
        w_new = w_int.reshape((3,1)) + w_dot.reshape((3,1)) * dt_int
        # print(w_new)

        omega_new = omega_matrix(-w_new)
        q_dot = .5 * omega @  q
        q_new = q_int + q_dot * dt_int # TODO try replacing with quaternion multiplication
        # print(f"q_int: {q_int}")
        # print(f"q_dot*dt: {q_dot*dt}")
        # q_new = quat_mult(q_int, q_dot * dt_int)
        # print(f"new quat: {q_new}")
        R_int =  Rot.from_quat(q_new.T, scalar_first=True).as_matrix()
        # print(R_int)
        w_t_abs_est = R_int @ w_s_abs.reshape((3,1)) - w_new.reshape((3,1))
        w_dot = R_int @ w_dot_s - np.linalg.inv(J_target) @ ( -np.cross(w_t_abs_est.reshape(3,),  (J_target @ w_t_abs_est).reshape(3,) )  ) - np.cross(w_t_abs_est.reshape(3,), w_new.reshape(3,))
        # TEMP - hardcoding 
        w_dot = np.zeros((3,1))

        w_int = w_new
        q_int = q_new

    


    # w_predict = state[9:12].reshape((3,1)) + w_dot.reshape((3,1)) * dt
    # q_predict = q + q_dot * dt

    # print(w_int)
    w_predict = w_int 
    q_predict = q_int

    # do nothing with the error MRP for now

    new_state = np.vstack((roe_predict.reshape((6,1)), np.zeros((3,1)), w_predict))

    return new_state, q_predict


def load_all_sq_params(param_path):

    all_files = [ f for f in os.listdir(param_path)]

    all_files.sort(key=lambda x: int(x[6:-11])) # from https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort

    print(all_files)

    params_all = np.zeros((len(all_files), 19*8))
    # print(params_all)
    for i in range(0, len(all_files)):
    # for f in all_files:
        f = all_files[i]
        # if os.path.isfile(f):
        #     print("File found!")
        params_dict = np.load(os.path.join(param_path, f), allow_pickle=True).item()
        # print(params_dict)

        params_list = params_dict.params
        # print(params_list)
        # params_vec = np.array([])

        # params_vec.append(params_list[0].cpu().numpy()[0].flatten())
        # params_vec.append(params_list[1].cpu().numpy()[0].flatten())
        # params_vec.append(params_list[2].cpu().numpy()[0].flatten())
        # params_vec.append(params_list[3].cpu().numpy()[0].flatten())
        # params_vec.append(params_list[5].cpu().numpy()[0].flatten())


        params_all[i, 0:8*3] = params_list[0].cpu().numpy()[0].flatten()
        params_all[i, 8*3:8*3+8*2] = params_list[1].cpu().numpy()[0].flatten()
        params_all[i, 8*3+8*2:8*3+8*2+8*3] = params_list[2].cpu().numpy()[0].flatten()
        params_all[i, 8*3+8*2+8*3:8*3+8*2+8*3+8*9] = params_list[3].cpu().numpy()[0].flatten()
        params_all[i, 8*3+8*2+8*3+8*9:8*3+8*2+8*3+8*9+8*2] = params_list[5].cpu().numpy()[0].flatten()
       
        # print(params_all)
        # params_vec.append(params_list[2].cpu().numpy()[0].flatten())
        # params_vec.append(params_list[3].cpu().numpy()[0].flatten())
        # params_vec.append(params_list[5].cpu().numpy()[0].flatten())


        # params_vec = np.array([params_dict['_shape'],
        #                        params_dict['_translation'],
        #                        params_dict['_rotation'],
        #                        params_dict['_taper']])
        # params_vec = np.flatten(params_dict.params.numpy())
        # print(params_vec)
        # params_all[i,:] = params_vec
        # print(params_all)
        

    # params_all = np.array(params_all)

    print(f"shape of params_all: {params_all.shape}")
    # print(f"sample params: {params_all[0]}")

    cov = np.cov(params_all.T).T
    # print(cov.shape)

    return params_all, all_files, cov


def reconstruct_sq_params(sq_vector):

    sq_vector = sq_vector.flatten()

    size = torch.from_numpy(sq_vector[ 0:8*3].reshape(1, 8, 3))
    shape = torch.from_numpy(sq_vector[ 8*3:8*3+8*2].reshape(1, 8, 2))
    translation = torch.from_numpy(sq_vector[8*3+8*2:8*3+8*2+8*3].reshape(1, 8, 3))
    rotation = torch.from_numpy(sq_vector[ 8*3+8*2+8*3:8*3+8*2+8*3+8*9].reshape(1, 8, 3,3))
    prob = torch.from_numpy(np.ones((8,)).reshape(1, 8, 1))
    taper =  torch.from_numpy(sq_vector[ 8*3+8*2+8*3+8*9:8*3+8*2+8*3+8*9+8*2].reshape(1, 8, 2))

    params = PrimitiveParameters(shape, size, translation, rotation, prob, taper)

    return params

def plot_sq(params_vec, fname):
    params_recon = reconstruct_sq_params(params_vec)
    # plot the reconstructed params as mesh
    # net = Model(cfg, 1200)
    mesh,pcls = net._convert_params_to_mesh(params_recon, train=False, combine_sq_to_one_mesh=False)

    # print(mesh[0])
    mesh = mesh[0]
    mesh_pr = []
    for i in range(8):
        # mesh_pr.append(
        #     Meshes(
        #         verts=[pcls[0, :, i]],
        #         faces=[net.mesh_converter.faces]
        #     ).cpu()
        # )
        mesh_pr.append(mesh[i])

    plot_3dmesh(mesh_pr, markers_for_vertices=False, savefn=fname)


    
def test_sq_funcs():
    sq_params_dir = "kf_params/test_1"

    all_sq_params, fnames, cov = load_all_sq_params(sq_params_dir)

    # compare reconstructed to original
    # print(all_sq_params.shape)
    params_recon = reconstruct_sq_params(all_sq_params[0,:])

    print(params_recon)

    # plot the reconstructed params as mesh
    
    mesh,pcls = net._convert_params_to_mesh(params_recon, train=False, combine_sq_to_one_mesh=False)

    # print(mesh[0])
    mesh = mesh[0]
    mesh_pr = []
    for i in range(8):
        # mesh_pr.append(
        #     Meshes(
        #         verts=[pcls[0, :, i]],
        #         faces=[net.mesh_converter.faces]
        #     ).cpu()
        # )
        mesh_pr.append(mesh[i])

    plot_3dmesh(mesh_pr, markers_for_vertices=False, savefn=str( f"test_mesh.jpg"))

    # print(fnames[0])

def roe_to_rel_pos(chief_oe, roe):

    # Based on AA 279D Lec 5/6 slide 8
    # linearized mapping of ROE to target postion w.r.t. chief in RTN frame

    a_c = chief_oe[0]
    e_c = chief_oe[1]
    i_c = chief_oe[2]
    raan_c = chief_oe[3]
    argp_c = chief_oe[4]
    f_c = chief_oe[5]

    da = roe[0]
    dl = roe[1]
    dex = roe[2]
    dey = roe[3]
    dix = roe[4]
    diy = roe[5]

    # Assume circular orbit to approximate M ~= f
    u = f_c + argp_c

    ex = e_c * np.cos(argp_c)
    ey = e_c * np.sin(argp_c)

    x_bar = da - dex * np.cos(u) - dey*np.sin(u)
    y_bar = dl + 2*dex*np.sin(u) - 2*dey*np.cos(u)
    z_bar = dix*np.sin(u) - diy*np.cos(u)

    # assume |r| = a
    return np.array([[a_c*x_bar, a_c*y_bar, a_c*z_bar]]).T




def main():

    # print(f"scipy version:{sp.__version__}")

    gt_dataset_path = "../datasets/shirtv1/roe1/roe1.json"
    gt_im_files, gt_rot_all, gt_trans_all = load_dataset_poses(gt_dataset_path)

    gt_metadata_path = "../datasets/shirtv1/roe1/metadata.json"
    gt_w_all, gt_oe_all, gt_roe_all, gt_w_s_abs_all, q_eci2pri_all, rv_all = load_dataset_metadata(gt_metadata_path)

    a_0 = gt_oe_all[0,0] / 1000 # km
    gt_roe_all = gt_roe_all / (a_0 * 1000)


    R_rtn_cam = determine_RTN_rot(rv_all[0,:], q_eci2pri_all[0,:])
    print(f"R: {R_rtn_cam}")

    # print(gt_roe_all)

    gt_pose_all = np.hstack((gt_rot_all, gt_trans_all))

    gt_state_all = np.hstack((gt_roe_all, gt_rot_all, gt_w_all)) # TODO figure out how to handle the rotation
    
    kf_version = "basic"
    reject_outliers = False

    # gt_pcl = torch.from_numpy(np.load("../spe3r_v1_1/cygnss_solo_39/surface_points.npz")["points"])
    gt_surface = np.load(
            "../spe3r_v1_1/cygnss_solo_39/surface_points.npz",
            allow_pickle=True
        )
    pidx = random.sample(list(range(100000)), 2500)
    gt_pcl = torch.tensor(gt_surface["points"][0, pidx], dtype=torch.float32).unsqueeze(0)



    sq_params_dir = "kf_params/test_1"
    sq_params_all, sq_fnames, sq_cov = load_all_sq_params(sq_params_dir)


    if kf_version == "basic":
        # our measurements are the gt poses w/o noise
        # plus some noisy shape estimates

        # TEMP: just handle poses at first

        meas_pose_all = gt_pose_all.copy()

    n_steps = len(gt_im_files)
    dt = 5
    n_sim = 200

    J_target = np.array([[2.6857031250000003,   0,   0], 
                         [ 0,   3.45807272329802,   0], 
                         [ 0, 0, 3.106463735035311]])
    J_servicer = np.array([[  16.704476666666668,  0,  0],
                           [  0,  19.440250000000002,  0],
                           [  0,  0,  18.278726666666664]])

    # get chief mean motion
    mu_earth = 3.986012e5 # km^3/m^2
    
    # print(a_0)
    n_c = np.sqrt(mu_earth / a_0**3) 
    print(f"n: {n_c}")

    Q_roe = np.eye(6)*1e-15
    Q_rot = np.eye(3)*2 # refers to error MRP vector
    Q_w = .5 * np.eye(3)
    Q_state = sp.linalg.block_diag(Q_roe, Q_rot, Q_w)
    # print(Q_state.shape)
    # print(Q_roe)

    R_rot = np.eye(4)*.5
    # R_rot = np.zeros(((4,4)))
    R_trans = np.eye(3)*2
    R_meas = sp.linalg.block_diag(R_rot, R_trans)

    roe_stm = np.eye(6)
    roe_stm[1,0] = -1.5*n_c*dt

    # print(roe_stm)

    # print(f"beginning gt roe: {gt_roe_all[0,:]}")

    Q_init = Q_state.copy() * 3
    Q_init[9:12,9:12] = np.eye(3)*.0001
    Q_init[0:6,0:6] = np.eye(6)*1e-14

    mu_prior = np.hstack((gt_roe_all[0,:], np.zeros((3,)), gt_w_all[0,:])).reshape((12,1)) + np.random.multivariate_normal(np.zeros((12,)), Q_init).reshape((12,1))
    # mu_prior = np.hstack((gt_roe_all[0,:], np.zeros((3,)), gt_w_all[0,:])).reshape((12,1))
    mu_q_prior = quat_mult(Rot.from_mrp(mu_prior[6:9].flatten()).as_quat(scalar_first=True) , gt_rot_all[0,:].reshape((4,1)) )
    mu_prior[6:9] = 0

    mu_sq_prior = sq_params_all[0,:].reshape((152,1))
    S_sq_prior = sq_cov * 2

    R_sq = sq_cov*20
    Q_sq = sq_cov/4
    
    print(mu_q_prior)
    
    print(gt_rot_all[0,:])
    S_prior = 3*Q_state

    # print(mu_prior[0]*a_0*1000)

    print(f"q prior: {mu_q_prior}")

    mu_predict_all = []
    S_predict_all = []
    q_predict_all = []
    sq_predict_all = []
    S_sq_predict_all = []

    mu_post_all = []
    S_post_all = []
    q_post_all = []
    sq_post_all = []
    S_sq_post_all = []
    chamfer_all = []
    chamfer_meas_all = []

    # for i in range(1, n_steps):
    for i in range(1,n_sim):
    # for i in range(1,2):

        # if i % 100 == 0:
        #     print(f"Iteration # {i}")
        print(f"Iteration # {i}")

        # meas_i = meas_pose_all[i,:]

        # from ground truth
        w_s_abs = np.array(gt_w_s_abs_all[i,:]).reshape((3,1))

        # Predict step
        # Get sigma points

        
        # print(f"S_prior: {np.diagonal(S_prior)}")
        # print(S_prior)
        # print(np.linalg.eig(S_prior))

        points, weights = unscented_transform(mu_prior, S_prior, 1)

        points_predict = []
        qs_predict = []
        sq_predict = []
        for pt in points:
            pt_predict, q_pred = predict_state(pt, mu_q_prior.reshape(4,), roe_stm, J_target, J_servicer, w_s_abs, dt)
            
            # pt_predict[6:9] = Rot.from_quat(q_pred, scalar_first=True).as_mrp().reshape((3,1))


            points_predict.append(pt_predict)
            qs_predict.append(q_pred)


        
        mu_predict, S_predict = inverse_unscented_transform(np.array(points_predict), np.array(weights).reshape(1,25))
        # Add a tiny bit of noise to w because we can't observe the dynamics
        # mu_predict[9:12] = mu_predict[9:12] + np.random.multivariate_normal(np.zeros((3,)), np.eye(3)*.00000001).reshape((3,1))
        
        S_predict = S_predict  + Q_state
        q_predict = np.array(weights).reshape((1,25)) @ np.array(qs_predict).reshape((25,4))


        # Re-normalize quaternion and set MRP back to 0
        q_predict = q_predict / (np.linalg.norm(q_predict))
        mu_predict[6:9] = 0

        mu_sq_predict = mu_sq_prior.reshape((152,1)) + np.random.multivariate_normal(np.zeros(152,), Q_sq/15).reshape((152,1))
        # mu_sq_predict[0:8*2+8*3] = np.abs(mu_sq_predict[0:8*2+8*3])
        S_sq_predict = S_sq_prior + Q_sq


        # print(f"S predict: {np.diagonal(S_predict)}")

        mu_predict_all.append(mu_predict)
        S_predict_all.append(S_predict)
        q_predict_all.append(q_predict)


        sq_predict_all.append(mu_sq_predict)
        S_sq_predict_all.append(S_sq_predict)

        # Measurement update

        # Stand-in for NN eval
        y_meas = gt_pose_all[i,:] + np.random.multivariate_normal(np.zeros(7,), R_meas)
        # TEMP: noiseless measurement
        # y_meas = gt_pose_all[i,:]
        
        # re-normalize quaternion estimate
        y_meas[0:4] = y_meas[0:4] / np.linalg.norm(y_meas[0:4])
        # print(f"Real pose: {gt_pose_all[i,:]}")
        # print(f"measurement: {y_meas}")
        # print(f"measured translation: {y_meas[4:]}")


        points, weights = unscented_transform(mu_predict, S_predict, 1)

        points_meas = []
        qs_post = []
        for pt in points:

            # get expected translation 
            t_expect = R_rtn_cam @ roe_to_rel_pos(gt_oe_all[i,:], pt[0:6])
            # print(f"Sigma point expected translation: {t_expect}")

            # get expected rotation
            pt_rot_err = Rot.from_mrp(pt[6:9]).as_quat(scalar_first=True).reshape((4,))
            # print(f"sigma point rotation error: {pt_rot_err}")
            # print(f"q predict: {q_predict}")
            q_expect = quat_mult(pt_rot_err, q_predict.reshape((4,))).reshape((4,1))

            y_expect = np.vstack((q_expect, t_expect))

            # print(f"y expect: {y_expect}")

            points_meas.append(y_expect)


        y_expect_mean, S_cov = inverse_unscented_transform(points_meas, weights)
        S_cov = S_cov + R_meas*3
        S_cc = cross_cov(points, points_meas, weights)

        # print(f"mean expected measurement: {y_expect_mean.T}")

        mu_post = mu_predict.reshape((12,1)) + ( S_cc @ np.linalg.inv(S_cov) @ (y_meas.reshape((7,1)) - y_expect_mean.reshape((7,1)))).reshape((12,1))
        S_post = S_predict - S_cc @ np.linalg.inv(S_cov) @ S_cc.T 

        # update quaternion
        q_err_post = Rot.from_mrp(mu_post[6:9].reshape(3,)).as_quat(scalar_first=True)
        q_post = quat_mult( q_err_post.reshape(4,), q_predict.reshape(4,))
        mu_post[6:9] = 0


        # SHape update
        sq_meas = sq_params_all[i,:]

        # Outlier detection - get the mahalanobis distance
        if reject_outliers:
            m_dist = sp.spatial.distance.mahalanobis(sq_meas.flatten(), mu_sq_predict.flatten(), np.linalg.inv(S_sq_predict))
        else:
            m_dist = 0
        # print(f"M distance: {m_dist}")

        if reject_outliers:
            reject = (i > 20) & (m_dist > 10)
        else:
            reject = False

        if not reject:
            # print((sq_meas).shape)
            mu_sq_post = mu_sq_predict.reshape((152,1)) + (S_sq_predict @ np.linalg.inv(S_sq_predict + R_sq) @ (sq_meas.reshape((152,1)) - mu_sq_predict.reshape((152,1)))).reshape((152,1))
            # mu_sq_post[0:8*2+8*3] = np.abs(mu_sq_post[0:8*2+8*3])
            S_sq_post = S_sq_predict - S_sq_predict @ np.linalg.inv(S_sq_predict + R_sq) @ S_sq_predict

            sq_post_all.append(mu_sq_post)
            S_sq_post_all.append(S_sq_post)
        else:
            print("Outlier rejected!")
            mu_sq_post = mu_sq_predict.reshape((152,1))
            S_sq_post = S_sq_predict - S_sq_predict @ np.linalg.inv(S_sq_predict + 15*R_sq) @ S_sq_predict

        # save latest shape estimate & compute chamfer distance
        # plot_sq(mu_sq_post, f"kf_figs/shape_ests/shape_{i}.png")
        mesh,pcls = net._convert_params_to_mesh(reconstruct_sq_params(mu_sq_post), train=False, combine_sq_to_one_mesh=False)
        # print(mesh[0].verts_padded().shape)
        # print(pcls.shape)

        all_verts = torch.zeros((1, 8*642, 3))
        mesh_verts = mesh[0].verts_padded()
        for j in range(8):
            # print(mesh_verts[j,:,:].shape)
            # print((all_verts[0,j*642:(j+1)*642,:]).shape)
            all_verts[0,j*642:(j+1)*642,:] = mesh_verts[j,:,:]
        cham,_ = chamfer_distance(all_verts, gt_pcl, batch_reduction='mean')
        chamfer_all.append(cham)

        # also compute for measurement
        mesh,_ = net._convert_params_to_mesh(reconstruct_sq_params(sq_meas), train=False, combine_sq_to_one_mesh=False)
        all_verts_meas = torch.zeros((1, 8*642, 3))
        mesh_verts_meas = mesh[0].verts_padded()
        for j in range(8):
            all_verts_meas[0,j*642:(j+1)*642,:] = mesh_verts_meas[j,:,:]
        cham_meas,_ = chamfer_distance(all_verts_meas, gt_pcl, batch_reduction='mean')
        chamfer_meas_all.append(cham_meas)          



        

        
        # print(f"mu_post: {mu_post.T}")
        # print(f"q post: {q_post}")
        # print(f"True state: {gt_state_all[i]}")

        # save values
        mu_post_all.append(mu_post)
        S_post_all.append(S_post)
        q_post_all.append(q_post)


        # postrior becomes prior
        mu_prior = mu_post
        mu_q_prior = q_post
        S_prior = S_post
        mu_sq_prior = mu_sq_post
        S_sq_prior = S_sq_post

        # print(mu_sq_post.shape)
        # print(f"estimated size params:")
        # print(mu_sq_post[0:8*3])

        # # postrior becomes prior
        # mu_prior = mu_predict
        # mu_q_prior = q_predict
        # S_prior = S_predict







    mu_predict_all = np.array(mu_predict_all)
    S_predict_all = np.array(S_predict_all)
    q_predict_all = np.array(q_predict_all)

    r,c,_ = mu_predict_all.shape
    mu_predict_all = mu_predict_all.reshape((r,c))
    q_predict_all = q_predict_all.reshape((r,4))

    mu_post_all = np.array(mu_post_all)
    S_post_all = np.array(S_post_all)
    q_post_all = np.array(q_post_all)
    # # TEMP
    # mu_post_all = np.zeros((r,c))
    # S_post_all = np.zeros((3,3,r))
    # q_post_all = np.zeros((r,4))

    mu_post_all = mu_post_all.reshape((r,c))
    q_post_all = q_post_all.reshape((r,4))


    chamfer_all = np.array(chamfer_all)
    chamfer_meas_all = np.array(chamfer_meas_all)

    # print(chamfer_all)
    # print(chamfer_meas_all)



    


    # Plot results

    time_vec = np.arange(1, n_steps)*dt # TODO update to include first step (used to initialize)


    fig_roe, ((ax_a, ax_l), (ax_ex, ax_ey), (ax_ix, ax_iy)) = plt.subplots(nrows=3, ncols=2)

    # print(mu_predict_all[10,0]*a_0*1000)
    # print(mu_predict_all.shape)
    
    ax_a.plot(time_vec[1:n_sim], (a_0 * 1000) * gt_roe_all[1:n_sim,0].flatten())
    # ax_a.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_predict_all[:,0].flatten(), label="Predicted")
    ax_a.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_post_all[:,0].flatten())
    ax_a.fill_between(time_vec[1:n_sim], 
                      (a_0 * 1000) * mu_post_all[:,0].flatten() - (a_0 * 1000)*np.sqrt(S_post_all[:,0,0]), 
                      (a_0 * 1000) * mu_post_all[:,0].flatten() + (a_0 * 1000)*np.sqrt(S_post_all[:,0,0]),
                      color='orange',
                      alpha=0.5)
    ax_a.set_xlabel("Time, s")
    ax_a.set_ylabel("a da, m")

    ax_l.plot(time_vec[1:n_sim], (a_0 * 1000) * gt_roe_all[1:n_sim,1].flatten())
    # ax_l.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_predict_all[:,1].flatten(), label="Predicted")
    ax_l.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_post_all[:,1].flatten())
    ax_l.fill_between(time_vec[1:n_sim], 
                      (a_0 * 1000) * mu_post_all[:,1].flatten() - (a_0 * 1000)*np.sqrt(S_post_all[:,1,1]), 
                      (a_0 * 1000) * mu_post_all[:,1].flatten() + (a_0 * 1000)*np.sqrt(S_post_all[:,1,1]),
                      color='orange',
                      alpha=0.5)
    ax_l.set_xlabel("Time, s")
    ax_l.set_ylabel("a dlambda, m")

    ax_ex.plot(time_vec[1:n_sim], (a_0 * 1000) * gt_roe_all[1:n_sim,2].flatten())
    # ax_ex.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_predict_all[:,2].flatten(), label="Predicted")
    ax_ex.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_post_all[:,2].flatten())
    ax_ex.fill_between(time_vec[1:n_sim], 
                      (a_0 * 1000) * mu_post_all[:,2].flatten() - (a_0 * 1000)*np.sqrt(S_post_all[:,2,2]), 
                      (a_0 * 1000) * mu_post_all[:,2].flatten() + (a_0 * 1000)*np.sqrt(S_post_all[:,2,2]),
                      color='orange',
                      alpha=0.5)
    ax_ex.set_xlabel("Time, s")
    ax_ex.set_ylabel("a de_x, m")

    ax_ey.plot(time_vec[1:n_sim], (a_0 * 1000) * gt_roe_all[1:n_sim,3].flatten())
    # ax_ey.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_predict_all[:,3].flatten(), label="Predicted")
    ax_ey.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_post_all[:,3].flatten())
    ax_ey.fill_between(time_vec[1:n_sim], 
                      (a_0 * 1000) * mu_post_all[:,3].flatten() - (a_0 * 1000)*np.sqrt(S_post_all[:,3,3]), 
                      (a_0 * 1000) * mu_post_all[:,3].flatten() + (a_0 * 1000)*np.sqrt(S_post_all[:,3,3]),
                      color='orange',
                      alpha=0.5)
    ax_ey.set_xlabel("Time, s")
    ax_ey.set_ylabel("a de_y, m")

    ax_ix.plot(time_vec[1:n_sim], (a_0 * 1000) * gt_roe_all[1:n_sim,4].flatten())
    # ax_ix.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_predict_all[:,4].flatten(), label="Predicted")
    ax_ix.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_post_all[:,4].flatten())
    ax_ix.fill_between(time_vec[1:n_sim], 
                      (a_0 * 1000) * mu_post_all[:,4].flatten() - (a_0 * 1000)*np.sqrt(S_post_all[:,4,4]), 
                      (a_0 * 1000) * mu_post_all[:,4].flatten() + (a_0 * 1000)*np.sqrt(S_post_all[:,4,4]),
                      color='orange',
                      alpha=0.5)
    ax_ix.set_xlabel("Time, s")
    ax_ix.set_ylabel("a di_x, m")

    ax_iy.plot(time_vec[1:n_sim], (a_0 * 1000) * gt_roe_all[1:n_sim,5].flatten(), label="Ground truth")
    # ax_iy.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_predict_all[:,5].flatten(), label="Predicted")
    ax_iy.plot(time_vec[1:n_sim], (a_0 * 1000) * mu_post_all[:,5].flatten(), label="Posterior")
    ax_iy.fill_between(time_vec[1:n_sim], 
                      (a_0 * 1000) * mu_post_all[:,5].flatten() - (a_0 * 1000)*np.sqrt(S_post_all[:,5,5]), 
                      (a_0 * 1000) * mu_post_all[:,5].flatten() + (a_0 * 1000)*np.sqrt(S_post_all[:,5,5]),
                      color='orange',
                      alpha=0.5,
                      label='1-sigma covariance')
    ax_iy.set_xlabel("Time, s")
    ax_iy.set_ylabel("a di_y, m")

    # plt.legend()


    # ax_iy.legend()
    plt.tight_layout()
    fig_roe.legend(loc='outside lower right')
    

    fig_roe.savefig('kf_figs/roe_plot.png')

    # compute the upper and lower bounds of angle estimate
    q_upper = []
    q_lower = []
    rot_err = []
    rot_err_cov = []
    for i in range(0,n_sim-1):
        dp = np.diagonal(S_post_all[i, 6:9, 6:9]).copy()
        # print(dp)
        dq = Rot.from_mrp(dp).as_quat(scalar_first=True)
        dq_inv = Rot.inv(Rot.from_mrp(dp)).as_quat(scalar_first=True)
        
        # print(q_post_all[i])
        # print(quat_mult(dq, q_post_all[i]))
        # print(quat_mult(-dq, q_post_all[i]))
        q_upper.append(-quat_mult(q_post_all[i], dq))
        q_lower.append(-quat_mult( dq_inv , q_post_all[i]))

        # get rotation  error
        R_gt = Rot.from_quat(gt_rot_all[i], scalar_first=True).as_matrix()
        R_est = Rot.from_quat(q_post_all[i], scalar_first=True).as_matrix()
        R_ub = Rot.from_quat(-quat_mult(q_post_all[i], dq), scalar_first=True).as_matrix()
        R_err = R_gt.T @ R_est
        R_err_ub = R_gt.T @ R_ub
        err_mag = Rot.from_matrix(R_err).magnitude()
        err_mag_cov = Rot.from_matrix(R_err_ub).magnitude()

        rot_err.append(err_mag)
        rot_err_cov.append(err_mag_cov)


    q_upper = np.array(q_upper)
    q_lower = np.array(q_lower)
    rot_err = np.array(rot_err)
    rot_err_cov = np.array(rot_err_cov)
    

    # print(q_upper[0].shape)



    fig_q, ((ax_q1, ax_q2), (ax_q3, ax_q4)) = plt.subplots(nrows=2, ncols=2)

    ax_q1.plot(time_vec[1:n_sim], gt_rot_all[1:n_sim,0])
    # ax_q1.plot(time_vec[1:n_sim], q_predict_all[0:,0])
    ax_q1.plot(time_vec[1:n_sim], q_post_all[0:,0])
    ax_q1.fill_between(time_vec[1:n_sim], 
                      q_upper[0:,0], 
                      q_lower[0:,0],
                      color='orange',
                      alpha=0.5)
    ax_q1.set_xlabel("Time, s")
    ax_q1.set_ylabel("q_w")

    ax_q2.plot(time_vec[1:n_sim],    gt_rot_all[1:n_sim,1])
    # ax_q2.plot(time_vec[1:n_sim], q_predict_all[0:,1])
    ax_q2.plot(time_vec[1:n_sim], q_post_all[0:,1])
    ax_q2.fill_between(time_vec[1:n_sim], 
                      q_upper[0:,1], 
                      q_lower[0:,1],
                      color='orange',
                      alpha=0.5)
    ax_q2.set_xlabel("Time, s")
    ax_q2.set_ylabel("q_x")

    ax_q3.plot(time_vec[1:n_sim],    gt_rot_all[1:n_sim,2])
    # ax_q3.plot(time_vec[1:n_sim], q_predict_all[0:,2])
    ax_q3.plot(time_vec[1:n_sim], q_post_all[0:,2])
    ax_q3.fill_between(time_vec[1:n_sim], 
                      q_upper[0:,2], 
                      q_lower[0:,2],
                      color='orange',
                      alpha=0.5)
    ax_q3.set_xlabel("Time, s")
    ax_q3.set_ylabel("q_y")

    ax_q4.plot(time_vec[1:n_sim],    gt_rot_all[1:n_sim,3], label="Ground truth")
    # ax_q4.plot(time_vec[1:n_sim], q_predict_all[0:,3], label="Predicted")
    ax_q4.plot(time_vec[1:n_sim], q_post_all[0:,3], label="Posterior")
    ax_q4.fill_between(time_vec[1:n_sim], 
                      q_upper[0:,3], 
                      q_lower[0:,3],
                      color='orange',
                      alpha=0.5,
                      label='1-sigma covariance')
    ax_q4.set_xlabel("Time, s")
    ax_q4.set_ylabel("q_z")

    # ax_q4.legend()
    fig_q.legend(loc='outside upper right')

    plt.tight_layout()



    fig_q.savefig('kf_figs/quat_plot.png')

    fig_re, ax_re = plt.subplots()

    ax_re.plot(time_vec[1:n_sim], np.rad2deg(rot_err), label="Rotation error", color="#ff7f0e")
    ax_re.fill_between(time_vec[1:n_sim], np.rad2deg(rot_err_cov), label="1-sigma covariance", color='orange',
                      alpha=0.5)
    
    ax_re.set_xlabel("Time, s")
    ax_re.set_ylabel("Rotation error, deg")

    fig_re.legend(loc='outside upper right')
    plt.tight_layout()

    fig_re.savefig('kf_figs/roterr_plot.png')

    


    fig_w, (ax_w1, ax_w2, ax_w3) = plt.subplots(nrows=3)

    ax_w1.plot(time_vec[1:n_sim], gt_w_all[1:n_sim,0].flatten())
    # ax_w1.plot(time_vec[1:n_sim],  mu_predict_all[:,9].flatten(), label="Predicted")
    ax_w1.plot(time_vec[1:n_sim],  mu_post_all[:,9].flatten())
    ax_w1.fill_between(time_vec[1:n_sim], 
                      mu_post_all[:,9].flatten() - np.sqrt(S_post_all[:,9,9]), 
                      mu_post_all[:,9].flatten() + np.sqrt(S_post_all[:,9,9]),
                      color='orange',
                      alpha=0.5)
    ax_w1.set_xlabel("Time, s")
    ax_w1.set_ylabel("w_1")

    ax_w2.plot(time_vec[1:n_sim], gt_w_all[1:n_sim,1].flatten())
    # ax_w2.plot(time_vec[1:n_sim],  mu_predict_all[:,10].flatten(), label="Predicted")
    ax_w2.plot(time_vec[1:n_sim],  mu_post_all[:,10].flatten())
    ax_w2.fill_between(time_vec[1:n_sim], 
                      mu_post_all[:,10].flatten() - np.sqrt(S_post_all[:,10,10]), 
                      mu_post_all[:,10].flatten() + np.sqrt(S_post_all[:,10,10]),
                      color='orange',
                      alpha=0.5)
    ax_w2.set_xlabel("Time, s")
    ax_w2.set_ylabel("w_2")

    ax_w3.plot(time_vec[1:n_sim], gt_w_all[1:n_sim,2].flatten(), label="Ground truth")
    # ax_w3.plot(time_vec[1:n_sim],  mu_predict_all[:,11].flatten(), label="Predicted")
    ax_w3.plot(time_vec[1:n_sim],  mu_post_all[:,11].flatten(), label="Posterior")
    ax_w3.fill_between(time_vec[1:n_sim], 
                      mu_post_all[:,11].flatten() - np.sqrt(S_post_all[:,11,11]), 
                      mu_post_all[:,11].flatten() + np.sqrt(S_post_all[:,11,11]),
                      color='orange',
                      alpha=0.5,
                      label='1-sigma covariance')
    ax_w3.set_xlabel("Time, s")
    ax_w3.set_ylabel("w_3")

    # ax_w3.legend()
    fig_w.legend(loc='outside upper right')
    plt.tight_layout()

    fig_w.savefig('kf_figs/angvel_plot.png')
    


    fig_s, ax_s = plt.subplots()
    ax_s.plot(time_vec[1:n_sim], chamfer_meas_all[0:n_sim].flatten(), label="Measurement chamfer distance")
    ax_s.plot(time_vec[1:n_sim], chamfer_all[0:n_sim].flatten(), label="Estimate chamfer distance")
    ax_s.set_xlabel("Time, s")
    ax_s.set_ylabel("Chamfer distance")
    ax_s.legend()
    # plt.tight_layout()

    fig_s.savefig('kf_figs/chamfer_plot.png')



def plot_data():
    pass


if __name__=="__main__":
    main()
    # test_sq_funcs()