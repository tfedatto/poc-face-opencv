import base64
import time

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.face_utils import FACIAL_LANDMARKS_68_IDXS, FACIAL_LANDMARKS_5_IDXS

from draw_face import draw
import reference_world as world
from scipy.spatial import distance as dist

shape_predictor = "shape_predictor_68_face_landmarks.dat"
imagePath = "face-closed-eyes/closed.jpeg"


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def check_closed_eyes(image, shp):
    EYE_AR_THRESH = 0.25
    shape = face_utils.shape_to_np(shp)
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # extract the left and right eye coordinates, then use the
    # coordinates to compute the eye aspect ratio for both eyes
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    # check to see if the eye aspect ratio is below the blink
    # threshold, and if so, increment the blink frame counter
    if ear < EYE_AR_THRESH:
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(image, "Closed eye", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        raise ValueError('Closed Eye')


def check_eyes_position(image, shp):
    shape = face_utils.shape_to_np(shp)

    if len(shape) == 68:
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    else:
        (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]

    # determine the scale of the new resulting image by taking
    # the ratio of the distance between eyes in the *current*
    # image to the ratio of distance between eyes in the
    # *desired* image
    dist = int(np.sqrt((dX ** 2) + (dY ** 2)))

    if dist in range(60, 90):
        return "Photo ok"
    elif dist < 60:
        raise ValueError("Photo looks too far away")
    elif dist > 90:
        raise ValueError("Photo seems to be very close")
    else:
        raise ValueError("Photo is not cool")


def get_shape(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)

    faces = detector(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 0)
    total_faces = len(faces)

    if total_faces > 1:
        raise ValueError(f'Only one face is needed. Detected {total_faces} faces')

    face = faces[0]
    return predictor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), face)


def check_head_pose(image, shape):
    focal = 1  # Callibrated Focal Length of the camera
    # Get refs points 2d image
    ref_img_pts = world.ref_2d_image_points(shape)
    face_3d_model = world.ref_3d_model()
    height, width, channel = image.shape
    focal_length = focal * width
    camera_matrix = world.camera_matrix(focal_length, (height / 2, width / 2))

    mdists = np.zeros((4, 1), dtype=np.float64)

    # calculate rotation and translation vector using solvePnP
    success, rotation_vector, translation_vector = cv2.solvePnP(
        face_3d_model, ref_img_pts, camera_matrix, mdists)

    nose_end_points_3d = np.array([[0, 0, 1000.0]], dtype=np.float64)
    nose_end_point_2d, jacobian = cv2.projectPoints(
        nose_end_points_3d, rotation_vector, translation_vector, camera_matrix, mdists)

    # draw nose line
    p1 = (int(ref_img_pts[0, 0]), int(ref_img_pts[0, 1]))
    p2 = (int(nose_end_point_2d[0, 0, 0]), int(nose_end_point_2d[0, 0, 1]))
    cv2.line(im, p1, p2, (110, 220, 0),
             thickness=2, lineType=cv2.LINE_AA)

    # calculating angle
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    pose = False
    gaze = "Looking: "
    if angles[1] < -15:
        gaze += "Left"
        pose = True
    elif angles[1] > 15:
        gaze += "Right"
        pose = True
    else:
        gaze += "Forward"

    cv2.putText(image, gaze, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 80), 2)
    if pose:
        raise ValueError(gaze)


if __name__ == "__main__":
    inicio = time.time()
    try:
        im = cv2.imread(imagePath)
        im = imutils.resize(im, width=300)
        clone = im.copy()
        restval, buffer = cv2.imencode(".jpg", clone)
        jpg_base64 = base64.b64encode(buffer)
        # print(jpg_base64)

        shape = get_shape(im)
        # Add face lines
        draw(im, shape)

        check_head_pose(im, shape)
        check_closed_eyes(im, shape)
        check_eyes_position(im, shape)
    except Exception as e:
        print(e)

    fim = time.time()
    tempo = (fim - inicio) * 1000

    print(f'O tempo de execução do calculo do fatorial foi de: {tempo:.2f} ms')
