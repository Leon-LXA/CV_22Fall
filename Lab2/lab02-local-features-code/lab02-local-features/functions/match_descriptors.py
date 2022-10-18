import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    q1 = desc1.shape[0]
    q2 = desc2.shape[0]
    # print(q1, q2)
    distances = np.zeros((q1, q2))
    for i in range(q1):
        for j in range(q2):
            distances[i][j] = np.sum((desc1[i] - desc2[j]) ** 2)

    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    # print(distances)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = np.empty((0, 2), int)
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        for i in range(q1):
            matches = np.row_stack((matches, np.array([i, np.argmin(distances[i])])))

    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        matches1to2 = np.empty((0, 2), int)
        matches2to1 = np.empty((0, 2), int)
        # img1到img2计算一次配对矩阵
        for i in range(q1):
            matches1to2 = np.row_stack((matches1to2, np.array([i, np.argmin(distances[i])])))
        # img2到img1反过来计算一次配对矩阵
        distancesT = distances.T
        for i in range(q2):
            matches2to1 = np.row_stack((matches2to1, np.array([i, np.argmin(distancesT[i])])))
        # print(matches1to2)
        # print(matches2to1)
        # 如果两个配对矩阵中对应的点是双射的话，就加入到返回的matches矩阵
        for i in range(q1):
            if matches2to1[matches1to2[i][1]][1] == i:
                matches = np.row_stack((matches, np.array([i, matches1to2[i][1]])))


    elif method == "ratio":
        # TODO: implement the ratio test matching here
        sort_distances = np.sort(distances, axis=1)
        for i in range(q1):
            if np.min(distances[i]) <= ratio_thresh * np.partition(sort_distances, kth=1, axis=1)[i][1]:
                matches = np.row_stack((matches, np.array([i, np.argmin(distances[i])])))

    else:
        raise NotImplementedError
    return matches

