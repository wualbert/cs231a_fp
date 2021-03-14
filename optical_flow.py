import numpy as np
import cv2
import point_clound_utils.io as io
import param
import point_clound_utils.visualize as visualize


def draw_tracks(frame_num, frame, mask, points_prev, points_curr, color):
    """Draw the tracks and create an image.
    """
    for i, (p_prev, p_curr) in enumerate(zip(points_prev, points_curr)):
        a, b = p_curr.ravel()
        c, d = p_prev.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(
            frame, (a, b), 3, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imwrite('frame_%d.png'%frame_num,img)
    return img


def compute_tracked_features_and_tranformation(frames, depths,
                                               intrinsic):
    """Code for question 5a.

    Output:
      p0, p1, p2: (N,2) list of numpy arrays representing the pixel coordinates of the
      tracked features.  Include the visualization and your answer to the
      questions in the separate PDF.
    """
    assert len(frames)==2
    # params for ShiTomasi corner detection
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(60,60),
        maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01),
        flags=(cv2.OPTFLOW_LK_GET_MIN_EIGENVALS))

    # Convert to gray images.
    old_frame = frames[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create some random colors for drawing
    color = np.random.randint(0, 255, (200, 3))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frames[1])
    track_p = [p0]
    prev_frame = old_gray
    prev_p = p0
    for i,frame in enumerate(frames[1:]):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # TODO: Fill in this code
        # BEGIN YOUR CODE HERE
        p, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, frame_gray, prev_p,
                                                None, **lk_params)
        print(np.average(err))
        track_p.append(p)
        points_curr = p[st==1]
        points_prev = prev_p[st==1]
        #Once you compute the new feature points for this frame, comment this out
        #to save images for your PDF:
        draw_tracks(i+1, frame, mask, points_prev, points_curr, color)
        prev_frame = frame_gray
        prev_p = p
        # END YOUR CODE HERE
    assert(len(track_p) == len(frames))
    track_p = np.squeeze(track_p)
    # return np.squeeze(p0), np.squeeze(track_p[0]), np.squeeze(track_p[1])
    # track_p is len(frame)*maxCorners(200)*2
    # Convert to homogeneous
    shape = np.array(track_p.shape)
    shape[2] = 1
    track_p_hom = np.append(track_p,np.ones(shape), axis=2)
    # Convert to 3D
    kinv = np.linalg.pinv(intrinsic)
    track_p_3d = (track_p_hom @ kinv.T)
    depth_values = []
    all_in_bound_indices = set(list(range(200)))
    for track_p_i in range(len(track_p)):
        in_bound_indices = np.where(np.logical_and(
            np.logical_and(track_p[track_p_i,:,1].astype(int)<480,
                           track_p[track_p_i,:,1].astype(int)>=0),
            np.logical_and(track_p[track_p_i, :, 0].astype(int) <640,
                           track_p[track_p_i, :, 0].astype(int) >= 0)))[0]
        all_in_bound_indices.intersection_update(set(in_bound_indices))
        depth_values.append(depths[track_p_i][np.minimum(track_p[track_p_i,:,1].astype(int),479,
                                                         dtype=int),
                                              np.minimum(track_p[track_p_i,:,0].astype(int),639,
                                                         dtype=int)])
    all_in_bound_indices = sorted(list(all_in_bound_indices))
    track_p_3d = track_p_3d[:,all_in_bound_indices,:]
    depth_values = np.asarray(depth_values)[:,all_in_bound_indices]
    in_bound_indices = np.where(np.all(np.logical_not(np.isclose(depth_values, 0)), axis=0))[0]
    # Discard invalid indices
    depth_values = depth_values[:,in_bound_indices]
    track_p_3d = track_p_3d[:,in_bound_indices,:]
    scales = np.divide(track_p_3d[:,:,2], depth_values)
    for i in range(3):
        track_p_3d[:,:,i] = np.divide(track_p_3d[:,:,i], scales)
    np.testing.assert_allclose(track_p_3d[:,:,-1], depth_values)

    # Solve for the transformations between each images
    Rs = []
    centroids = np.average(track_p_3d, axis=1)
    Ts = centroids[1:,:] - centroids[:-1,:]
    track_p_3d_normalized = np.copy(track_p_3d)
    for i in range(track_p_3d_normalized.shape[0]):
        track_p_3d_normalized[i,:,:] -= centroids[i,:]
    for i in range(1, len(images)):
        X = track_p_3d_normalized[i-1,:]
        Y = track_p_3d_normalized[i, :]
        A = X.T@Y
        U, S, VT = np.linalg.svd(A)
        M = np.eye(3)
        M[2,2] = np.linalg.det(VT.T@U.T)
        R = VT.T@M@U.T
        Rs.append(R)
    Rs = np.atleast_3d(np.asarray(Rs))
    assert(len(Rs)==len(Ts))
    return Rs, Ts


if __name__ == "__main__":
    images = []
    depths = []
    image_numbers = [8,9]
    for i in image_numbers:
        image, depth = io.load_rgb_and_d(i)
        images.append(image)
        depths.append(depth)
    intrinsic = io.load_camera_intrinsics_open3d().intrinsic_matrix
    RTs = compute_tracked_features_and_tranformation(images, depths,
                                                            intrinsic)
    transformation= np.eye(4)
    transformation[:3,:3] = RTs[0][0]
    transformation[:3,-1] = RTs[1][0]

    # Visualize point clouds
    pcd1 = io.load_cloud_from_selected_image(image_number=image_numbers[0])
    pcd2 = io.load_cloud_from_selected_image(image_number=image_numbers[1])
    print('Trans', transformation)
    visualize.draw_registration_result_open3d(pcd1, pcd2, transformation, [pcd1])

