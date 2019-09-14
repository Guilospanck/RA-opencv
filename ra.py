import cv2
import sys
import numpy as np

# Constants
MIN_MATCHES = 15

def drawKeypoints(figure, figure_kps):
    """
    Draw keypoints over a figure.

    Parameters:
    - @figure: What image you want to plot the keypoints.
    - @figure_kps: Keypoints of this image.
    """
    # draw only keypoints location, not size and orientation
    figure_with_kps = cv2.drawKeypoints(figure, figure_kps, figure_with_kps, color=(0,255,0), flags=0)
    cv2.imshow('Keypoints', figure_with_kps)
    cv2.waitKey(0)

def findHomography(src_keypoints, dst_keypoints, matches):
    """
    Calculates and returns the Homography matrix between two planes: the train image plane and the query image plane.
    Homography is a 3x3 matrix that correlates the points that exist in two planes.

    Parameters:
    - @src_keypoints: Keypoints from the train image.
    - @dst_keypoints: Keypoints from the destiny image.
    - @matches: Matches between the two sets (train and query images).
    """

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

def drawRectangle(source, src_kps, dst_kps, matches, destiny):
    """
    Uses the shape of the train image, along with the Homography previously calculated, to 'send' the points
    from one plane (train image plane) to another (query image plane).

    Parameters:
    - @source: it is the train image, to retrieve its shape (height and width).
    - @src_kps: Keypoints from the train image.
    - @dst_kps: Keypoints from the destiny image.
    - @matches: Matches found between the train and query images.
    - @destiny: it is the query image, in which the rectangle is going to be drawn.
    """

    h, w = source.shape # height and width of the image that we are searching
    
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2) # reshape(rows, cols, dimension)

    # get Homography
    M = findHomography(src_kps, dst_kps, matches)

    # project corners into the frame
    dst = cv2.perspectiveTransform(pts, M)

    # connect them with lines
    img_with_lines_connected = cv2.polylines(destiny, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    cv2.imshow('Connected', img_with_lines_connected)
    cv2.waitKey(0)

def run():

    source = cv2.imread('marley_source.jpg', 0)
    destiny = cv2.imread('marley_destiny.jpg', 0)
    # colordestiny = cv2.imread('marley_destiny.jpg', cv2.IMREAD_COLOR)
    # frame = cv2.VideoCapture(0)

    # Initiate orb detector
    orb = cv2.ORB_create()

    # Create BFMatcher
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    # Find the keypoints with orb
    source_kps = orb.detect(source, None) # 500 keypoints
    destiny_kps = orb.detect(destiny, None)

    # compute the descriptors with orb
    source_kps, source_des = orb.compute(source, source_kps)  # 500 descriptors with 32 integer values each
    destiny_kps, destiny_des = orb.compute(destiny, destiny_kps)

    # Match destiny with source descriptors
    matches = bf.match(source_des, destiny_des)

    # Sort matches based on their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Verifiy if the number of matches satisfy a min threshold
    if (len(matches) > MIN_MATCHES):
        # draw first 15 matches
        fig = cv2.drawMatches(source, source_kps, destiny, destiny_kps, matches[:MIN_MATCHES], 0, flags=2)

        # show results
        cv2.imshow('Matches', fig)
        cv2.waitKey(0)

        drawRectangle(source, source_kps, destiny_kps, matches, destiny)
    else:
       print("Not enough matches have been found - %d/%d " % (len(matches), MIN_MATCHES))

if __name__ == '__main__':
    run()
