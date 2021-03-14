import param
import point_clound_utils.io as io
import point_clound_utils.visualize as visualize
import numpy as np
import copy
import open3d as o3d

def eval_segpose_predictions(image_ids=None,
                             object_names=None):
    if image_ids is None:
        image_ids = param.selected_linemod_image_ids
    if object_names is None:
        object_names = list(param.selected_linemod_objects.keys())
    model_clouds = io.load_clouds_from_selected_models(True, object_names)
    d_average = np.zeros((len(object_names),len(image_ids)))
    d_sd = np.zeros((len(object_names),len(image_ids)))
    for obj_i, object_name in enumerate(object_names):
        for img_i, image_id in enumerate(image_ids):
            scene_cloud = io.load_cloud_from_selected_image(image_id)
            RT_gt = np.eye(4)
            RT_segpose = np.eye(4)
            RT_gt[:3,:] = io.load_ground_truth_pose(object_name,image_id)
            RT_segpose[:3,:] = io.load_segpose_prediction(object_name,image_id)
            cloud_gt = copy.copy(model_clouds[object_name])
            cloud_gt.transform(RT_gt)
            cloud_segpose = copy.copy(model_clouds[object_name])
            cloud_segpose.transform(RT_segpose)
            d_gt = np.asarray(cloud_segpose.compute_point_cloud_distance(cloud_gt))
            d_average[obj_i, img_i] = np.average(d_gt)
            d_sd[obj_i, img_i] = np.std(d_gt)
            print(RT_gt,'gt', np.linalg.inv(RT_gt))
            print(RT_segpose, 'seg')
            visualize.draw_point_clouds([cloud_gt, cloud_segpose, scene_cloud],
                                        [[1,0,0],[0,0,1]])
            # visualize.draw_registration_result_open3d(cloud_gt, cloud_segpose, np.eye(4),
            #                                           other_clouds=[scene_cloud])
    return d_average, d_sd

def eval_segpose_icp_predictions(image_ids=None,
                             object_names=None):
    if image_ids is None:
        image_ids = param.selected_linemod_image_ids
    if object_names is None:
        object_names = list(param.selected_linemod_objects.keys())
    model_clouds = io.load_clouds_from_selected_models(True, object_names)
    d_seg_average = np.zeros((len(object_names),len(image_ids)))
    d_seg_sd = np.zeros((len(object_names),len(image_ids)))

    d_est_average = np.zeros((len(object_names),len(image_ids)))
    d_est_sd = np.zeros((len(object_names),len(image_ids)))

    for obj_i, object_name in enumerate(object_names):
        for img_i, image_id in enumerate(image_ids):
            scene_cloud = io.load_cloud_from_selected_image(image_id)
            RT_gt = np.eye(4)
            RT_segpose = np.eye(4)
            RT_gt[:3,:] = io.load_ground_truth_pose(object_name,image_id)
            RT_segpose[:3,:] = io.load_segpose_prediction(object_name,image_id)
            # compute ICP using segpose prediction as the seed
            threshold = 0.1
            reg_p2p = o3d.pipelines.registration.registration_icp(
                model_clouds[object_name], scene_cloud, threshold, RT_segpose)
            RT_est = reg_p2p.transformation
            model_cloud_gt = copy.copy(model_clouds[object_name])
            model_cloud_gt.transform(RT_gt)
            model_cloud_seg = copy.copy(model_clouds[object_name])
            model_cloud_seg.transform(RT_segpose)
            model_cloud_est = copy.copy(model_clouds[object_name])
            model_cloud_est.transform(RT_est)

            d_seg = np.asarray(model_cloud_gt.compute_point_cloud_distance(model_cloud_seg))
            d_est = np.asarray(model_cloud_gt.compute_point_cloud_distance(model_cloud_est))
            d_seg_average[obj_i, img_i] = np.average(d_seg)
            d_seg_sd[obj_i, img_i] = np.std(d_seg)
            d_est_average[obj_i, img_i] = np.average(d_est)
            d_est_sd[obj_i, img_i] = np.std(d_est)
            # visualize.draw_point_clouds([model_cloud_gt, model_cloud_seg, model_cloud_est, scene_cloud],
            #                             [[1,0,0],[0,1,0],[0,0,1]]) #R: GT, G: Seg, B: Est
    return d_seg_average, d_seg_sd, d_est_average, d_est_sd


if __name__ == "__main__":
    # results = eval_segpose_predictions(None, ['can'])
    results = eval_segpose_icp_predictions(None, ['can'])
    for i in range(len(results)//2):
        print(f'Error: {results[2*i]} +/- {results[2*i+1]}\n')

