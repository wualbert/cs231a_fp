# from utils import *
# from segpose_net import SegPoseNet
#
# class SegmentationDrivenPose:
#     def __init__(self):
#         data_cfg = './data/data-LINEMOD.cfg'
#         data_options = read_data_cfg(data_cfg)
#         self.m = SegPoseNet(data_options)
#
#     def do_detect(self, img, intrinsics):
#         predPose = do_detect(self.m, img, intrinsics, bestCnt=10, conf_thresh=0.3)
#         print(predPose)
#         return predPose