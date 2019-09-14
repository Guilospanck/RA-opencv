import cv2
import sys
import numpy as np

MIN_MATCHES = 15

scene = cv2.imread('scene.jpg', 0)
model = cv2.imread('model.jpg', 0)
# frame = cv2.VideoCapture(0)

# Initiate orb detector
orb = cv2.ORB_create()

# Create BFMatcher
bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

# Find the keypoints with orb
scene_kps = orb.detect(scene, None)
model_kps = orb.detect(model, None)

# compute the descriptors with orb
scene_kps, scene_des = orb.compute(scene, scene_kps)
model_kps, model_des = orb.compute(model, model_kps)

# Match model with scene descriptors
matches = bf.match(model_des, scene_des)

# Sort matches based on their distance
matches = sorted(matches, key=lambda x: x.distance)

# Verifiy if the number of matches satisfy a min threshold
if (len(matches) > MIN_MATCHES):
    # draw first 15 matches
    fig = cv2.drawMatches(scene, scene_kps, model, model_kps, matches, 0, flags=2)

    # show results
    cv2.imshow('Matches', fig)
    cv2.waitKey(0)
else:
    print("Not enough matches have been found - %d/%d " % (len(matches), MIN_MATCHES))

def drawKeypoints(figure, figure_kps):
    # draw only keypoints location, not size and orientation
    figure_with_kps = cv2.drawKeypoints(figure, figure, figure, color=(0,255,0), flags=0)
    cv2.imshow('Keypoints', figure_with_kps)
    cv2.waitKey(0)

def findHomography(src_keypoints, dst_keypoints, matches):

    # source points
    src_pts = np.float32([
        src_keypoints[m.queryIdx].pt for m in matches # points on source image in relation to the destination image (query image)
    ]).reshape(-1, 1, 2)

    # destination points
    dst_points = np.float32([
        dst_keypoints[m.trainIdx].pt for m in matches # points on the destination image in relation to the source image (train image)
    ]).reshape(-1, 1, 2)

    # compute Homography
    M, mask = cv2.findHomography(src_pts, dst_points, cv2.RANSAC, 5.0)

    return M

def drawRectangle(source, src_kps, dst_kps, matches, model):
    h, w = source.shape # height and width of the image that we are searching
    
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2) # reshape(rows, cols, dimension)

    # get Homography
    M = findHomography(src_kps, dst_kps, matches)

    # project corners into the frame
    dst = cv2.perspectiveTransform(pts, M)

    # connect them with lines
    img_with_lines_connected = cv2.polylines(model,[np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    cv2.imshow('Connected', img_with_lines_connected)
    cv2.waitKey(0)

drawRectangle(scene, scene_kps, model_kps, matches, model)
