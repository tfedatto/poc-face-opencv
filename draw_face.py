import cv2
import numpy as np


def draw_polyline(img, shapes, start, end, is_closed=False):
    points = []
    for i in range(start, end + 1):
        point = [shapes.part(i).x, shapes.part(i).y]
        points.append(point)
    points = np.array(points, dtype=np.int32)
    cv2.polylines(img, [points], is_closed, (255, 80, 0),
                  thickness=1, lineType=cv2.LINE_8)


def draw(img, shapes):
    draw_polyline(img, shapes, 0, 16)
    draw_polyline(img, shapes, 17, 21)
    draw_polyline(img, shapes, 22, 26)
    draw_polyline(img, shapes, 27, 30)
    draw_polyline(img, shapes, 30, 35, True)
    draw_polyline(img, shapes, 36, 41, True)
    draw_polyline(img, shapes, 42, 47, True)
    draw_polyline(img, shapes, 48, 59, True)
    draw_polyline(img, shapes, 60, 67, True)
