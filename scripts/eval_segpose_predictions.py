import param
import point_clound_utils.io as io
import point_clound_utils.visualize as visualize
import numpy as np
import copy

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
            visualize.draw_registration_result_open3d(cloud_gt, cloud_segpose, np.eye(4),
                                                      other_clouds=[scene_cloud])
    return d_average, d_sd

if __name__ == "__main__":
    results = eval_segpose_predictions(None, ['can'])
    print(results)

