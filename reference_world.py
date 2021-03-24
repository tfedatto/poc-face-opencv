import numpy as np


def ref_3d_model():
    modelPoints = [[0.0, 0.0, 0.0],           # Nose tip
                   [0.0, -330.0, -65.0],      # Chin
                   [-225.0, 170.0, -135.0],   # Left eye left corner
                   [225.0, 170.0, -135.0],    # Right eye right corner
                   [-150.0, -150.0, -125.0],  # Left Mouth corner
                   [150.0, -150.0, -125.0]]   # Right mouth corner
    return np.array(modelPoints, dtype=np.float64)


def ref_2d_image_points(shape):
    # To understand look at the image facial_landmarks_68markup-768x619.jpeg
    # The image contains the facial points and here we take the important points
    # Remember you are viewing a list, when looking at the image the value that is in the
    # index will correspond to the index + 1.
    # Example chin index 8 in the image corresponds to point 9
    imagePoints = [[shape.part(30).x, shape.part(30).y],  # Nose tip
                   [shape.part(8).x, shape.part(8).y],    # Chin
                   [shape.part(36).x, shape.part(36).y],  # Left eye left corner
                   [shape.part(45).x, shape.part(45).y],  # Right eye right corner
                   [shape.part(48).x, shape.part(48).y],  # Left Mouth corner
                   [shape.part(54).x, shape.part(54).y]]  # Right mouth corner
    return np.array(imagePoints, dtype=np.float64)


def camera_matrix(fl, center):
    mat = [[fl, 1, center[0]],
           [0, fl, center[1]],
           [0, 0, 1]]
    return np.array(mat, dtype=np.float)
