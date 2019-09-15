import cv2
import sys
import numpy as np
import math
import os
from objloader_simple import *

import argparse

# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 60

# Command line argument parsing
parser = argparse.ArgumentParser(description='Augmented Reality')

parser.add_argument('-r', '--rectangle', help= 'Draw rectangle delimiting target surface on frame.', action='store_true')
parser.add_argument('-mk', '--model_keypoints', help= 'Draw model keypoints.', action='store_true')
parser.add_argument('-fk', '--frame_keypoints', help= 'Draw frame keypoints.', action='store_true')
parser.add_argument('-ma', '--matches', help= 'Draw matches between keypoints.', action='store_true')

args = parser.parse_args()


def drawKeypoints(figure, figure_kps):
    """
    Draw keypoints over a figure.

    Parameters:
    - @figure: What image you want to plot the keypoints.
    - @figure_kps: Keypoints of this image.
    """
    # draw only keypoints location, not size and orientation
    figure = cv2.drawKeypoints(figure, figure_kps, figure, color=(0,255,0), flags=0)

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
    destiny = cv2.polylines(destiny, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

def intrinsic_camera(fu, fv, u0, v0):
    K = np.float32([fu, 0, u0, 0, fv, v0, 0, 0, 1])
    K = K.reshape(3, 3)
    return K

def projection_matrix(K, H):
    """
    From the camera calibration matrix (intrinsic matrix) and the estimated Homography,
    compute the 3D projection matrix.

    Parameters:
    - @K: Intrinsics Matrix
    - @H: Homography Matrix

    - We have that H = K [R1 R2 t]    where H = homography matrix and R3 is ommited because the Homography
    is between two planes (x, y, z=0).

    - So, to retrieve the [R1 R2 t], we can do:    G = [G1 G2 G3] = K^-1 * H    where G1 = R1, G2 = R2, G3 = t and K = calibration matrix

    - The external calibration matrix ( [R1 R2 R3 t] ) is a homogeneous transformation, so, [R1 R2 R3] have to
    be orthonormal (condition to be homogeneous). So, in theory, to discover R3 we might do R3 = R1 x R2. BUT...
        a) Since we obtained this values (R1 and R2) from an approximation (homography), we cannot guarantee that
            the vectors will be orthonormals.
        b) The problem, then, is to find two vectors R1' and R2' that are close to R1 and R2 and are orthonormals. By doing
            that, we will calculate the R3 by doing R3 = R1' x R2'.

    - So, first, to encounter 2 vectors that are orthonormal, we must normalize R1 and R2 ( G1 and G2 ):
        l = sqrt( ||G1|| * ||G2|| )
                        R1 = G1/l   R2 = G2/l    t = G3/l

        and find a vector that are orthonormal to either R1 and R2:

                        p = R1 x R2

        And then the new basis will be rotated approximately 45° clockwise, resulting in a new basis:   c and d where:

                c = R1 + R2    and d = c x p = (R1 + R2) x (R1 x R2)

        Now, if this new basis (c, d) is transformed into unit vectors, that is to say c' = c/||c|| and d' = d/||d||,
        and rotated 45° counterclockwise, we will have a basis that is orthonormal and is pretty close to the 
        real values of R1 and R2. So:

                R1' = 1/sqrt(2) * (c/||c||  +  d/||d||)    rotated 45° counterclockwise with values normalized
                R2' = 1/sqrt(2) * (c/||c||  -  d/||d||)                 ||

    - Now that we have our orthonormal basis that is close to the real R1 and R2, we can calculate the value of R3:

                    R3 = R1' x R2'
    
    - 3D projection matrix:      K * [R1' R2' R3 t]

    - For detailed explanation, open the image 'understanding/projection_matrix.png'.

    """

    # computes G = [G1 G2 G3] = K^-1 * H
    H = H * (-1) # If don't change this, the obj will render on the wrong axis. E.g.: if a change the book with a rotation forward, the correct was the obj go back, but, with normal H, the object actually shows the wrong face
    G = np.dot(np.linalg.inv(K), H)

    col1 = G[:,0]
    col2 = G[:,1]
    col3 = G[:,2]

    # normalize vectors
    l = math.sqrt(np.linalg.norm(col1, 2) * np.linalg.norm(col2, 2)) # l = sqrt( ||G1|| * ||G2|| )
    R1 = col1/l
    R2 = col2/l
    t = col3/l

    # compute the orthonormal basis 45° clockwise
    c = R1 + R2
    p = np.cross(R1, R2)
    d = np.cross(c, p)

    # compute the orthonormal basis 45° counterclockwise
    R1_line = np.dot(1 / math.sqrt(2), c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2)) # R1' = 1/sqrt(2) * ( c/||c|| + d/||d|| )
    R2_line = np.dot(1 / math.sqrt(2), c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2)) # R2' = 1/sqrt(2) * ( c/||c|| - d/||d|| )

    # compute R3
    R3 = np.cross(R1_line, R2_line)

    # compute the 3D projection matrix
    extrinsic = np.stack((R1_line, R2_line, R3, t)).T
    projection = np.dot(K, extrinsic)

    return projection

def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    hex_color = hex_color.lstrip('#')
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))

def render(img, obj, projection, model, color=False):
    """
    Render a simple Wavefront object (.obj) into the current video frame
    """

    vertices = obj.vertices
    scale_matrix = np.eye(3) * 3 # allow us to scale the model
    h, w = model.shape

    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.dot(points, scale_matrix)

        # render model in the middle of the reference surface. To do so, model points must be displaced.
        points = np.array([[p[0] + w / 2, p[1] + h / 2, p[2]] for p in points])
        dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
        imgpts = np.int32(dst)

        if color is False:
            cv2.fillConvexPoly(img, imgpts, (137, 27, 211))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1] # reverse
            cv2.fillConvexPoly(img, imgpts, color)
    
    return img

def run():

    # gets the current directory
    dir_name = os.getcwd()
    dir_name = dir_name.split('\\')
    dir_name.pop()
    dir_name = '\\'.join(dir_name)
    print(dir_name)

    # gets the train image
    source = cv2.imread(os.path.join(dir_name,'references/marley_source.jpg'), 0)

    # init video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Gets the Intrinsic matrix
    K = intrinsic_camera(fu=800, fv=800, u0=320, v0=240)

    # loads the 3d .obj model using the objoader_simple.py helper
    obj = OBJ(os.path.join(dir_name, 'models/fox/fox.obj'), swapyz=True)

    # Initiate orb detector
    orb = cv2.ORB_create()

    # Create BFMatcher
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    # Find the keypoints and descriptors of the train image with orb
    source_kps, source_des = orb.detectAndCompute(source, None) #  keypoints = 500 ; descriptors = 500 with 32 integer values each

    while True:
        # read the current frame
        ok, frame = cap.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
            return

        # find the keypoints and descriptors of the frame
        frame_kps, frame_des = orb.detectAndCompute(frame, None)

        # match frame descriptors with source descriptors
        matches = bf.match(source_des, frame_des)

        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)

        cv2.imshow('Frame', frame)

        # close
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # verify if enough matches are found. If yes, compute Homography
        if len(matches) > MIN_MATCHES:
            H = findHomography(source_kps, frame_kps, matches)

            # draw rectangle if there is this arg
            if args.rectangle:
                drawRectangle(source, source_kps, frame_kps, matches, frame)
            
            # if a valid homography was found, render the cube on model plane
            if H is not None:
                try:
                    # obtain the projection matrix
                    P = projection_matrix(K, H) 
                    # project cube or model
                    frame = render(frame, obj, P, source, False)                    
                except:
                    pass
            
            if args.matches:
                # draw the first MIN_MATCHES
                frame = cv2.drawMatches(source, source_kps, frame, frame_kps, matches[:MIN_MATCHES], 0, flags=2)
            
            # show the results
            cv2.imshow('Frame', frame)

            # close
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        else:
            print("Not enough matches found - %d/%d " % (len(matches), MIN_MATCHES))
    
    cap.release()
    cv2.destroyAllWindows()
    return 0
    

if __name__ == '__main__':
    run()
