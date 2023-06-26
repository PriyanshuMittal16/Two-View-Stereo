import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""
    Kiinv=np.linalg.inv(K_i)
    Kjinv=np.linalg.inv(K_j)
    gomo_i=np.matmul(K_i_corr, np.matmul(R_irect, Kiinv))
    gomo_j=np.matmul(K_j_corr, np.matmul(R_jrect, Kjinv))
    rgb_i_rect=cv2.warpPerspective(rgb_i, gomo_i, (w_max,h_max))
    rgb_j_rect=cv2.warpPerspective(rgb_j, gomo_j, (w_max, h_max))
        
    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_iw, R_jw : [3,3]
    T_iw, T_jw : [3,1]
        p_i = R_iw @ p_w + T_iw
        p_j = R_jw @ p_w + T_jw
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ij @ p_j + T_ij, B is the baseline
    """

    """Student Code Starts"""
    # p_j = R_jw @ p_w + T_jw
    I=np.array([0,0,0,1.0])
    HI=np.column_stack((R_iw,T_iw))
    HJ=np.column_stack((R_jw,T_jw))
    PI=np.vstack((HI, I))
    PJ=np.vstack((HJ,I))
    Pinvj=np.linalg.inv(PJ)

    PJI=np.matmul(PI, Pinvj)
    R_ij=PJI[:3, :3]
    T_ij=PJI[:3,3].reshape(3,1)
    # R_ij= np.matmul(R_jw.T, R_iw)
    # T_ij= -T_jw + T_iw
    ###--------------WHAT IS BASELINE________########

    # Bn=-np.matmul(R_ij.T, T_ij)
    B=np.linalg.norm(T_ij)
    """Student Code Ends"""
    return R_ij, T_ij, B


def compute_rectification_R(T_ij):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ij : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ij.squeeze(-1) / (T_ij.squeeze(-1)[1] + EPS)

    """Student Code Starts"""

    # r1= e_i/np.linalg.norm(e_i)
    # Z=np.array([0,0,1]) ####Check dimensions
    
    # A=np.cross(r1,Z)
    # r2=A/np.linalg.norm(A)

    # r3=np.cross(r2,r1)
    
    # R_irect=np.zeros((3,3))
    # R_irect[0]=r2.T
    # R_irect[1]=r1.T
    # R_irect[2]=r3.T

    F=T_ij.astype(np.float64)
    normF=np.linalg.norm(F)
    r2=(F/(normF+EPS))
    r2=r2.flatten()
    r1=np.array([F[1,0], -F[0,0], 0])
    r1f=(np.linalg.norm(r1)+EPS).flatten()
    r1=r1/r1f
    r3=np.cross(r1.flatten(), r2.flatten())
    

    """Student Code Ends"""
    R_irect = np.vstack((r1, r2, r3))

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    m=src.shape[0]
    d=dst.shape[0]
    ssd=np.zeros((m,d))
    distance =dst[np.newaxis,:,:,:]

    source=src[:,np.newaxis,:,:]
    

    Ghata=source-distance
    norm=np.linalg.norm(Ghata,axis=2)
    ssd=np.sum(norm**2, axis=2)

    
    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    distance =dst[np.newaxis,:,:,:]
    source=src[:,np.newaxis,:,:]
    Ghata=source-distance

    sad=np.abs(Ghata).sum(axis=3).sum(axis=2)
    
    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    msrc=src.mean(axis=1)
    mdst=dst.mean(axis=1)

    srcnorm=(src-msrc[:,np.newaxis,:])
    dstnorm=(dst-mdst[:,np.newaxis,:])

    ssigma=np.std(src, axis=1)
    dsigma=np.std(dst, axis=1)

    newns=srcnorm[:,np.newaxis,:,:]
    newnd=dstnorm[np.newaxis,:,:,:]
    newss=ssigma[:,np.newaxis,:]
    newsd=dsigma[np.newaxis,:,:]

    numerator=newns*newnd
    denominator=(newss*newsd)+EPS
    
    sumn=numerator.sum(axis=2)
    zncc=(sumn/denominator).sum(axis=2)
    """Student Code Ends"""

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    K=k_size**2

    n=k_size//2
    i0=image[:, :, 0].copy()
    i1=image[:, :, 1].copy()
    i2=image[:, :, 2].copy()

    m=image.shape[0]
    d=image.shape[1]

    r=np.pad(i0,n)
    g=np.pad(i1,n)
    b=np.pad(i2,n)

    pr=np.zeros((m,d, K))
    pg=np.zeros((m,d, K))
    pb=np.zeros((m,d, K))

    for i in range(n, m+n):
        for j in range(n, d+n):
            pr[i-n,j-n,:]=r[i-n:i+1+n, j-n:j+1+n].ravel()
            pg[i-n,j-n,:]=g[i-n:i+1+n, j-n:j+1+n].ravel()
            pb[i-n,j-n,:]=b[i-n:i+1+n, j-n:j+1+n].ravel()

    patch_buffer = np.stack((pr,pg,pb), axis=3)
        


    
    """Student Code Starts"""

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func : function, optional
        this is for auto-grader purpose, in grading, we will use our correct implementation 
        of the image2path function to exclude double count for errors in image2patch function

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    """Student Code Starts"""
    m,d=rgb_i.shape[:2]
    disp_map = np.zeros((m,d),dtype=np.float64)
    lr_consistency_mask = np.zeros((m,d),dtype=np.float64)
    ai,aj=np.arange(m), np.arange(m)

    cand=ai[:,None]-aj[None,:]+d0
    maskv=cand > 0.0
    count=np.arange(0,m,1)
    pi=image2patch(rgb_i.astype(float)/255.0, k_size)
    pj=image2patch(rgb_j.astype(float)/255.0, k_size)

    for i in tqdm(range(d)):
        pai, paj=pi[:,i], pj[:,i]
        epsilon=kernel_func(pai,paj)

        maxep=epsilon.max()+1.0
        epsilon[~maskv]=maxep

        minep=epsilon.argmin(axis=1)
        minepr=(epsilon[:,minep.flatten()]).argmin(axis=0)

        lr_consistency_mask[:,i]=((ai==minepr).flatten()).astype(float)
        disp_map[:,i]=count - minep +d0



    
    """Student Code Ends"""

    return disp_map, lr_consistency_mask

def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""
    div=B/disp_map
    dep_map=K[1,1]*div
    m=dep_map.shape[0]
    d=dep_map.shape[1]
    p=disp_map.shape[0]
    q=disp_map.shape[1]

    xyz_cam=np.zeros((m,d,3))

    ap=np.arange(p)
    aq=np.arange(q)

    x,y=np.meshgrid(aq,ap)

    x=x.flatten()
    y=y.flatten()
    de=dep_map.flatten()

    xc=(x-K[0,2])*de/K[0,0]
    yc=(y-K[1,2])*de/K[1,1]

    stack=np.stack([xc,yc,de]).T
    xyz_cam=stack.reshape((m,d,3))
    
    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_cw,
    T_cw,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""
    I=np.array([[0,0,0,1]])
    H=np.column_stack((R_cw,T_cw))
    P=np.vstack((H,I))
    P=np.linalg.inv(P)

    newR=P[0:3, 0:3]
    newT=P[0:3,-1]#--------------------------------------------------------------------------------
    
    A=np.matmul(newR, pcl_cam.T)
    pcl_world=(A+newT.reshape(3,1)).T
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_iw, T_iw = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_jw, T_jw = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ij, T_ij, B = compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw)
    assert T_ij[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ij)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ij,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_cw=R_irect @ R_iw,
        T_cw=R_irect @ T_iw,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
