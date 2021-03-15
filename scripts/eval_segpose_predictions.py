import param
import point_clound_utils.io as io
import point_clound_utils.visualize as visualize
import visualization_utils as vizu
import optical_flow
import numpy as np
import copy
from filterpy.kalman import ExtendedKalmanFilter
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import matplotlib.pyplot as plt
import os
import time
# def eval_segpose_predictions(image_ids=None,
#                              object_names=None):
#     if image_ids is None:
#         image_ids = param.selected_linemod_image_ids
#     if object_names is None:
#         object_names = list(param.selected_linemod_objects.keys())
#     model_clouds = io.load_clouds_from_selected_models(True, object_names)
#     d_average = np.zeros((len(object_names),len(image_ids)))
#     d_sd = np.zeros((len(object_names),len(image_ids)))
#     for obj_i, object_name in enumerate(object_names):
#         for img_i, image_id in enumerate(image_ids):
#             scene_cloud = io.load_cloud_from_selected_image_id(image_id)
#             RT_gt = np.eye(4)
#             RT_segpose = np.eye(4)
#             RT_gt[:3,:] = io.load_ground_truth_pose(object_name,image_id)
#             RT_segpose[:3,:] = io.load_segpose_prediction(object_name,image_id)
#             cloud_gt = copy.copy(model_clouds[object_name])
#             cloud_gt.transform(RT_gt)
#             cloud_segpose = copy.copy(model_clouds[object_name])
#             cloud_segpose.transform(RT_segpose)
#             d_gt = np.asarray(cloud_segpose.compute_point_cloud_distance(cloud_gt))
#             d_average[obj_i, img_i] = np.average(d_gt)
#             d_sd[obj_i, img_i] = np.std(d_gt)
#             print(RT_gt,'gt', np.linalg.inv(RT_gt))
#             print(RT_segpose, 'seg')
#             visualize.draw_point_clouds([cloud_gt, cloud_segpose, scene_cloud],
#                                         [[1,0,0],[0,0,1]])
#             # visualize.draw_registration_result_open3d(cloud_gt, cloud_segpose, np.eye(4),
#             #                                           other_clouds=[scene_cloud])
#     return d_average, d_sd
#
# def eval_segpose_icp_predictions(image_ids=None,
#                              object_names=None):
#     if image_ids is None:
#         image_ids = param.selected_linemod_image_ids
#     if object_names is None:
#         object_names = list(param.selected_linemod_objects.keys())
#     model_clouds = io.load_clouds_from_selected_models(True, object_names)
#     d_seg_average = np.zeros((len(object_names),len(image_ids)))
#     d_seg_sd = np.zeros((len(object_names),len(image_ids)))
#
#     d_est_average = np.zeros((len(object_names),len(image_ids)))
#     d_est_sd = np.zeros((len(object_names),len(image_ids)))
#
#     for obj_i, object_name in enumerate(object_names):
#         for img_i, image_id in enumerate(image_ids):
#             scene_cloud = io.load_cloud_from_selected_image_id(image_id)
#             RT_gt = np.eye(4)
#             RT_segpose = np.eye(4)
#             RT_gt[:3,:] = io.load_ground_truth_pose(object_name,image_id)
#             RT_segpose[:3,:] = io.load_segpose_prediction(object_name,image_id)
#             # compute ICP using segpose prediction as the seed
#             threshold = 0.1
#             reg_p2p = o3d.pipelines.registration.registration_icp(
#                 model_clouds[object_name], scene_cloud, threshold, RT_segpose)
#             RT_est = reg_p2p.transformation
#             model_cloud_gt = copy.copy(model_clouds[object_name])
#             model_cloud_gt.transform(RT_gt)
#             model_cloud_seg = copy.copy(model_clouds[object_name])
#             model_cloud_seg.transform(RT_segpose)
#             model_cloud_est = copy.copy(model_clouds[object_name])
#             model_cloud_est.transform(RT_est)
#
#             d_seg = np.asarray(model_cloud_gt.compute_point_cloud_distance(model_cloud_seg))
#             d_est = np.asarray(model_cloud_gt.compute_point_cloud_distance(model_cloud_est))
#             d_seg_average[obj_i, img_i] = np.average(d_seg)
#             d_seg_sd[obj_i, img_i] = np.std(d_seg)
#             d_est_average[obj_i, img_i] = np.average(d_est)
#             d_est_sd[obj_i, img_i] = np.std(d_est)
#             # visualize.draw_point_clouds([model_cloud_gt, model_cloud_seg, model_cloud_est, scene_cloud],
#             #                             [[1,0,0],[0,1,0],[0,0,1]]) #R: GT, G: Seg, B: Est
#     return d_seg_average, d_seg_sd, d_est_average, d_est_sd
#
# def eval_segpose_icp_kalman_predictions_old(image_ids=None,
#                              object_names=None):
#     if image_ids is None:
#         image_ids = param.selected_linemod_image_ids
#     if object_names is None:
#         object_names = list(param.selected_linemod_objects.keys())
#     model_clouds = io.load_clouds_from_selected_models(True, object_names)
#     d_seg_average = np.zeros((len(object_names),len(image_ids)))
#     d_seg_sd = np.zeros((len(object_names),len(image_ids)))
#
#     d_est_average = np.zeros((len(object_names),len(image_ids)))
#     d_est_sd = np.zeros((len(object_names),len(image_ids)))
#
#     images = []
#     depths = []
#     for i in image_ids:
#         image, depth = io.load_rgb_and_d(i)
#         images.append(image)
#         depths.append(depth)
#
#     # First compute the image to image transformations
#     intrinsic = io.load_camera_intrinsics_open3d().intrinsic_matrix
#     image_RTs = optical_flow.compute_tracked_features_and_tranformation(
#         images, depths, intrinsic)
#
#     def hx(x,*args):
#         return x
#     def H(x, *args):
#         return np.eye(6)
#     # Set up an Extended Kalman filter. Assume we are observing the 6D pose.
#     ekf = ExtendedKalmanFilter(6, 6) # xyz, euler
#     for obj_i, object_name in enumerate(object_names):
#         for img_i, image_id in enumerate(image_ids):
#             scene_cloud = io.load_cloud_from_selected_image_id(image_id)
#             RT_gt = np.eye(4)
#             RT_segpose = np.eye(4)
#             RT_gt[:3,:] = io.load_ground_truth_pose(object_name,image_id)
#             RT_segpose[:3,:] = io.load_segpose_prediction(object_name,image_id)
#             # compute ICP using segpose prediction as the seed
#             threshold = 0.1
#             reg_p2p = o3d.pipelines.registration.registration_icp(
#                 model_clouds[object_name], scene_cloud, threshold, RT_segpose)
#             RT_est = reg_p2p.transformation
#
#             # update Kalman filter
#             transformation = np.eye(4)
#             if img_i > 0:
#                 transformation[:3, :3] = image_RTs[0][img_i-1]
#                 transformation[:3, -1] = image_RTs[1][img_i-1]
#             # Do EKF as euler angles
#             euler_angles = R.from_matrix(RT_est[:3,:3]).as_euler('xyz')
#             state = np.hstack([RT_est[:3,-1], euler_angles])
#             print(state,'s')
#             ekf.predict_update(np.reshape(state,(-1,1)), H, hx)
#             state_post = np.ndarray.flatten(ekf.x_post)
#             print(state_post)
#             RT_kalman = np.eye(4)
#             RT_kalman[:3,:3] = R.from_euler('xyz',state_post[3:]).as_matrix()
#             RT_kalman[:3,-1] = state_post[:3]
#
#             model_cloud_gt = copy.copy(model_clouds[object_name])
#             model_cloud_gt.transform(RT_gt)
#             model_cloud_seg = copy.copy(model_clouds[object_name])
#             model_cloud_seg.transform(RT_segpose)
#             model_cloud_est = copy.copy(model_clouds[object_name])
#             model_cloud_est.transform(RT_est)
#             model_cloud_kalman = copy.copy(model_clouds[object_name])
#             model_cloud_kalman.transform(RT_kalman)
#             #
#             # d_seg = np.asarray(model_cloud_gt.compute_point_cloud_distance(model_cloud_seg))
#             # d_est = np.asarray(model_cloud_gt.compute_point_cloud_distance(model_cloud_est))
#             # d_seg_average[obj_i, img_i] = np.average(d_seg)
#             # d_seg_sd[obj_i, img_i] = np.std(d_seg)
#             # d_est_average[obj_i, img_i] = np.average(d_est)
#             # d_est_sd[obj_i, img_i] = np.std(d_est)
#             visualize.draw_point_clouds([model_cloud_gt, model_cloud_seg,
#                                          model_cloud_est, model_cloud_kalman,
#                                          scene_cloud],
#                                         [[1,0,0],[0,1,0],[0,0,1],
#                                          [1,1,0]]) #R: GT, G: Seg, B: Est, Y:Kalman
#     return d_seg_average, d_seg_sd, d_est_average, d_est_sd

def eval_segpose_icp_kalman_predictions(image_ids=None,
                             object_names=None,
                                        selected_noise=param.selected_noise):
    if image_ids is None:
        image_ids = param.selected_linemod_image_ids
    if object_names is None:
        object_names = list(param.selected_linemod_objects.keys())
    model_clouds = io.load_clouds_from_selected_models(True, object_names)

    ds_avg = np.zeros((len(object_names), len(image_ids), 5))
    ds_std = np.zeros((len(object_names), len(image_ids), 5))

    images = []
    depths = []
    for i in image_ids:
        image, depth = io.load_rgb_and_d(i,selected_noise=selected_noise)
        images.append(image)
        depths.append(depth)

    # First compute the image to image transformations
    intrinsic = io.load_camera_intrinsics_open3d().intrinsic_matrix
    ans = optical_flow.compute_tracked_features_and_tranformation(
        images, depths, intrinsic)
    image_RTs = ans[:2]
    err = ans[2]
    def hx(x,image_RT):
        x_shape = x.shape
        x = np.ndarray.flatten(x)
        RT_x = np.eye(4)
        RT_x[:3, :3] = R.from_euler('xyz', x[3:]).as_matrix()
        RT_x[:3, -1] = x[:3]
        x_pred = image_RT @ RT_x
        # Convert back to euler
        euler_angles = R.from_matrix(x_pred[:3, :3]).as_euler('xyz')
        state = np.hstack([x_pred[:3, -1], euler_angles])
        return np.reshape(state, x_shape)

    def H(x):
        return np.eye(6)

    # Set up an Extended Kalman filter. Assume we are observing the 6D pose.
    ekf = ExtendedKalmanFilter(6, 6) # xyz, euler
    prev_RT_kalman_input = None
    for obj_i, object_name in enumerate(object_names):
        for img_i, image_id in enumerate(image_ids):
            scene_cloud = io.load_cloud_from_selected_image_id(image_id,
                                                               selected_noise=selected_noise,
                                                               intrinsics=intrinsic)
            RT_gt = np.eye(4)
            RT_segpose = np.eye(4)
            RT_gt[:3,:] = io.load_ground_truth_pose(object_name,image_id)
            RT_segpose[:3,:] = io.load_segpose_prediction(object_name,image_id,
                                                          selected_noise=selected_noise)
            # compute ICP using segpose prediction as the seed
            threshold = 0.1
            reg_p2p = o3d.pipelines.registration.registration_icp(
                model_clouds[object_name], scene_cloud, threshold, RT_segpose)
            RT_est = reg_p2p.transformation

            # update Kalman filter
            RT_kalman_input = RT_segpose
            image_transformation = np.eye(4)
            if img_i > 0:
                image_transformation[:3, :3] = image_RTs[0][img_i-1]
                image_transformation[:3, -1] = image_RTs[1][img_i-1]

            if prev_RT_kalman_input is None:
                euler_angles = R.from_matrix(RT_kalman_input[:3, :3]).as_euler('xyz')
                state_observed = np.hstack([RT_kalman_input[:3, -1], euler_angles])
                ekf.predict_update(np.reshape(state_observed, (-1, 1)), H, hx,
                                   hx_args=(np.eye(4),))
            # Do EKF as euler angles
            # Compute the estimation
            euler_angles = R.from_matrix(RT_kalman_input[:3,:3]).as_euler('xyz')
            state_observed = np.hstack([RT_kalman_input[:3,-1], euler_angles])
            ekf.predict_update(np.reshape(state_observed,(-1,1)), H, hx,
                               hx_args=(image_transformation,))
            state_post = np.ndarray.flatten(ekf.x_post)
            RT_kalman = np.eye(4)
            RT_kalman[:3,:3] = R.from_euler('xyz',state_post[3:]).as_matrix()
            RT_kalman[:3,-1] = state_post[:3]

            state_prior = np.ndarray.flatten(ekf.x_prior)
            RT_kalman_prior = np.eye(4)
            RT_kalman_prior[:3,:3] = R.from_euler('xyz',state_prior[3:]).as_matrix()
            RT_kalman_prior[:3,-1] = state_prior[:3]

            reg_kalman = o3d.pipelines.registration.registration_icp(
                model_clouds[object_name], scene_cloud, threshold, RT_kalman)
            RT_kalman_icp = reg_kalman.transformation
            # Memoize the previous RT estimation from point cloud & segpose
            prev_RT_kalman_input = np.copy(RT_kalman_input)

            model_cloud_gt = copy.copy(model_clouds[object_name])
            model_cloud_gt.transform(RT_gt)
            model_cloud_seg = copy.copy(model_clouds[object_name])
            model_cloud_seg.transform(RT_segpose)
            model_cloud_est = copy.copy(model_clouds[object_name])
            model_cloud_est.transform(RT_est)
            model_cloud_kalman = copy.copy(model_clouds[object_name])
            model_cloud_kalman.transform(RT_kalman)
            model_cloud_kalman_icp = copy.copy(model_clouds[object_name])
            model_cloud_kalman_icp.transform(RT_kalman_icp)
            model_cloud_kalman_prior = copy.copy(model_clouds[object_name])
            model_cloud_kalman_prior.transform(RT_kalman_prior)

            print(img_i)
            for cloud_i, cloud in enumerate([model_cloud_seg,
                                             model_cloud_est,
                                             model_cloud_kalman,
                                             model_cloud_kalman_icp]):
                d_current = np.asarray(model_cloud_gt.compute_point_cloud_distance(cloud))
                ds_avg[obj_i, img_i, cloud_i] = np.average(d_current)
                ds_std[obj_i, img_i, cloud_i] = np.std(d_current)
            # visualize.draw_point_clouds([model_cloud_gt, model_cloud_seg,
            #                              model_cloud_est, model_cloud_kalman,
            #                              model_cloud_kalman_icp,
            #                              scene_cloud],
            #                             [[1,0,0],[0,1,0],[0,0,1],
            #                              [1,1,0],[1,0,1]]) #R: GT, G: Seg, B: Seg+ICP, Y:Kalman, P: Kalman+ICP
    return ds_avg, ds_std


if __name__ == "__main__":
    # results = eval_segpose_predictions(None, ['can'])
    # results = eval_segpose_icp_predictions(None, ['can'])
    timenow = str(time.time())
    objects = ['can']
    noises = 0.05
    ds, _ = eval_segpose_icp_kalman_predictions(None, objects,
                                                selected_noise=noises)
    drop_count = 3
    ds = ds[:,drop_count:,:]
    labels = ['Segpose', 'Segpose+ICP', 'Segpose+Kalman',
              'Segpose+Kalman+ICP']
    colors = ['r','g','b','y']
    os.mkdir(timenow)
    for obj_id, object in enumerate(objects):
        fig, ax = plt.subplots()
        for i in range(len(labels)):
            ax.scatter(np.arange(len(ds[obj_id])),
                       ds[obj_id, :, i], c=colors[i], alpha=0.7)
        ax.legend(labels)
        ax.set_title(f'Pose Estimation Error with Noise Amount {noises}')
        ax.set_xlabel('Image number')
        ax.set_ylabel('Error')
        plt.savefig(f'{timenow}/'+object + '.png', dpi=300)
    fig, ax = plt.subplots()
    ds = ds[0, :, :-1]
    ds_avg = np.average(ds,axis=0)
    ds_std = np.std(ds,axis=0)
    print(ds_avg, ds_std)
    ds_min = np.min(ds,axis=0)
    ds_max = np.max(ds,axis=0)
    # vizu.plot_error_with_min_max(range(len(ds_avg)), ds_avg, ds_std, ds_min,
    #                              ds_max,fig=fig, ax=ax,
    #                         label=labels)
    vizu.plot_error(range(len(ds_avg)), ds_avg, ds_std,fig=fig, ax=ax,label=labels)
    plt.savefig(f'{timenow}/error_bar.png', dpi=300)
    # for obj_id, object in enumerate(objects):
    #     fig, axs = plt.subplots()
    #     for i in range(len(labels)):
    #         for n in range(len(noises)):
    #             axs[n].scatter(np.arange(len(ds_all[n][obj_id])),
    #                        ds_all[n][obj_id,:,i],c=colors[i])
    #     axs[0].legend(labels)
    #     plt.show()