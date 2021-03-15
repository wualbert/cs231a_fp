selected_models = [1, 5, 10, 15, 17]
selected_noise = 0.05
linemod_path = '/Users/albertwu/exp/segmentation-driven-pose/data/OcclusionChallengeICCV2015/'
segpose_out_path_no_noise = '/Users/albertwu/exp/segmentation-driven-pose/Occluded-LINEMOD-Out'
segpose_out_path = f'/Users/albertwu/exp/segmentation-driven-pose/Occluded-LINEMOD-Out-{selected_noise}'


selected_linemod_objects = {'Driller':6, 'Holepuncher':10, 'Can':4,
                            'Glue':9}

# selected_linemod_image_ids = list(range(97,108))
# selected_linemod_image_ids = list(range(128, 135))
# selected_linemod_image_ids = list(range(5, 15))
# selected_linemod_image_ids = list(reversed(list(range(1025, 1036))))
# selected_linemod_image_ids = list(range(1019, 1036))
selected_linemod_image_ids = list(range(264, 276))