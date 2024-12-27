#!/usr/bin/python3
import cv2
import sys
import numpy as np
import math
import os
from objloader_simple import OBJ
import platform
import argparse
import time
from concurrent.futures import ThreadPoolExecutor


# Minimum number of matches that have to be found
# to consider the recognition valid
MIN_MATCHES = 120
# Only check for matches every some frames
FRAME_SKIP = 2
# Change this to the train image you want to find any match in the video stream
TRAIN_IMAGE_RELATIVE_PATH = "references/software.jpg"
# Change this to the .obj model you want to render on the matched image
OBJ_MODEL_RELATIVE_PATH = "models/fox/fox.obj"
# Change this to the .mtl model you want to render on the matched image
MTL_MODEL_RELATIVE_PATH = "models/fox/fox.mtl"
# Change this to increase/decrease the scale of the rendered .obj
MODEL_OBJ_SCALE = 3

# Command line argument parsing
parser = argparse.ArgumentParser(description="Augmented Reality")

parser.add_argument(
    "-r",
    "--rectangle",
    help="Draw rectangle delimiting target surface on frame.",
    action="store_true",
)
parser.add_argument(
    "-ma", "--matches", help="Draw matches between keypoints.", action="store_true"
)

args = parser.parse_args()


def benchmark(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} execution time: {end_time - start_time:.6f} seconds")
        return result

    return wrapper


@benchmark
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
    src_pts = np.float32(
        [
            src_keypoints[m.queryIdx].pt
            for m in matches  # points on source image in relation to the destination image (query image)
        ]
    ).reshape(-1, 1, 2)

    # destination points
    dst_points = np.float32(
        [
            dst_keypoints[m.trainIdx].pt
            for m in matches  # points on the destination image in relation to the source image (train image)
        ]
    ).reshape(-1, 1, 2)

    # compute Homography
    M, _ = cv2.findHomography(src_pts, dst_points, cv2.RANSAC, 5.0)

    return M


def drawRectangle(source, homography, destiny):
    """
    Uses the shape of the train image, along with the Homography previously calculated, to 'send' the points
    from one plane (train image plane) to another (query image plane).

    Parameters:
    - @source: it is the train image, to retrieve its shape (height and width).
    - @homography: homography matrix between source and query images.
    - @destiny: it is the query image, in which the rectangle is going to be drawn.
    """

    h, w = source.shape  # height and width of the image that we are searching

    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(
        -1, 1, 2
    )  # reshape(rows, cols, dimension)

    # project corners into the frame
    dst = cv2.perspectiveTransform(pts, homography)

    # connect them with lines
    destiny = cv2.polylines(destiny, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)


def intrinsic_camera(fu, fv, u0, v0):
    K = np.float32([fu, 0, u0, 0, fv, v0, 0, 0, 1])
    K = K.reshape(3, 3)
    return K


@benchmark
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
    H = H * (
        -1
    )  # If don't change this, the obj will render on the wrong axis. E.g.: if a change the book with a rotation forward, the correct was the obj go back, but, with normal H, the object actually shows the wrong face
    G = np.dot(np.linalg.inv(K), H)

    col1 = G[:, 0]
    col2 = G[:, 1]
    col3 = G[:, 2]

    # normalize vectors (uses geometric mean so no vector has more "weight" than the other)
    l = math.sqrt(
        np.linalg.norm(col1, 2) * np.linalg.norm(col2, 2)
    )  # l = sqrt( ||G1|| * ||G2|| )
    R1 = col1 / l
    R2 = col2 / l
    t = col3 / l

    # compute the orthonormal basis 45° clockwise
    c = R1 + R2
    p = np.cross(R1, R2)
    d = np.cross(c, p)

    # compute the orthonormal basis 45° counterclockwise
    R1_line = np.dot(
        1 / math.sqrt(2), c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2)
    )  # R1' = 1/sqrt(2) * ( c/||c|| + d/||d|| )
    R2_line = np.dot(
        1 / math.sqrt(2), c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2)
    )  # R2' = 1/sqrt(2) * ( c/||c|| - d/||d|| )

    # compute R3
    R3 = np.cross(R1_line, R2_line)

    # compute the 3D projection matrix
    extrinsic = np.stack((R1_line, R2_line, R3, t)).T
    projection = np.dot(K, extrinsic)

    return projection


def preprocess_face(face_data):
    """Precompute transformations for a single face."""
    (
        face,
        tex_ids,
        vertices,
        texcoords,
        projection,
        obj,
        material_name,
        scale_matrix,
        source_h,
        source_w,
    ) = face_data

    # 3D points for the current face
    points = np.array([vertices[vertex] for vertex in face])
    points = np.dot(points, scale_matrix)

    # Center the model on the reference surface
    points = np.array(
        [[p[0] + source_w / 2, p[1] + source_h / 2, p[2]] for p in points]
    )
    dst = cv2.perspectiveTransform(points.reshape(-1, 1, 3), projection)
    imgpts = np.int32(dst)

    # Get texture for the current material
    texture = obj.mtl.materials[material_name].get("texture", None)

    if texture is not None:
        # Map texture coordinates to the face
        uv = texcoords[tex_ids]
        uv[:, 0] *= texture.shape[1]
        uv[:, 1] *= texture.shape[0]

        # Compute the affine transform matrix
        src_pts = np.float32(uv[:3]).reshape(-1, 2)  # First 3 UV points
        dst_pts = np.float32(imgpts[:3]).reshape(-1, 2)  # First 3 projected points
        matrix = cv2.getAffineTransform(src_pts, dst_pts)

        return {"imgpts": imgpts, "matrix": matrix, "texture": texture}
    else:
        # Render with solid color
        color = obj.get_material_color(material_name)
        return {"imgpts": imgpts, "color": [int(c * 255) for c in color]}


@benchmark
def render(
    *,
    frame,
    obj,
    projection,
    vertices,
    texcoords,
    scale_matrix,
    source_h,
    source_w,
):
    """
    Render a Wavefront object (.obj) into the current video frame with texture mapping.
    Args:
        frame: The current video frame.
        obj: The parsed OBJ file.
        projection: Projection matrix (3x4: intrinsic + extrinsic).
        vertices: the obj vertices in NDArray[floating[_32Bit]] format.
        texcoords: the obj texture coordinates in NDArray[floating[_32Bit]] format.
        scale_matrix: the matrix to be used to scale the obj model in NDArray[floating[_32Bit]] format.
        source_h: the train image height.
        source_w: the train image width.
    Returns:
        frame: The rendered image with the object overlaid.
    """

    # Precompute face data for parallel processing
    face_data_list = [
        (
            face,
            tex_ids,
            vertices,
            texcoords,
            projection,
            obj,
            material_name,
            scale_matrix,
            source_h,
            source_w,
        )
        for face, tex_ids, _, material_name in obj.faces
    ]

    # Preprocess faces in parallel
    with ThreadPoolExecutor() as executor:
        processed_faces = list(executor.map(preprocess_face, face_data_list))

    # Combine all faces sequentially
    for face_data in processed_faces:
        imgpts = face_data["imgpts"]

        if "texture" in face_data:
            texture = face_data["texture"]
            matrix = face_data["matrix"]

            # Create a local triangular mask
            local_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

            # Warp texture and mask
            warped_texture = cv2.warpAffine(
                texture, matrix, (frame.shape[1], frame.shape[0])
            )
            cv2.fillConvexPoly(local_mask, imgpts[:3], 255)

            # Restrict operations to bounding box
            x, y, w, h = cv2.boundingRect(imgpts)
            frame[y : y + h, x : x + w] = cv2.copyTo(
                warped_texture[y : y + h, x : x + w],
                local_mask[y : y + h, x : x + w, None],
                frame[y : y + h, x : x + w],
            )

        else:
            # Render with solid color
            color = face_data["color"]
            cv2.fillConvexPoly(frame, imgpts, color=color)

    return frame


def getVideoCaptureSettingBasedOnPlatform():
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    elif system == "Linux":
        return cv2.CAP_V4L2
    else:
        print(f"Unknown operating system: {system}")
        return None


def run():
    # gets the current directory
    dir_name = os.getcwd()
    dir_name = dir_name.split("\\")
    dir_name.pop()
    dir_name = "\\".join(dir_name)

    # gets the train image
    source = cv2.imread(os.path.join(dir_name, TRAIN_IMAGE_RELATIVE_PATH), 0)

    # init video capture
    cap = cv2.VideoCapture(0, getVideoCaptureSettingBasedOnPlatform())
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Gets the Intrinsic matrix
    K = intrinsic_camera(fu=950, fv=950, u0=640, v0=360)

    # loads the 3d .obj model using the objoader_simple.py helper
    obj_filename = os.path.join(dir_name, OBJ_MODEL_RELATIVE_PATH)
    mtl_filename = os.path.join(dir_name, MTL_MODEL_RELATIVE_PATH)
    obj = OBJ(obj_filename, mtl_filename, swapyz=True)

    # Initiate orb detector
    orb = cv2.ORB_create()

    # Create BFMatcher
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    # Find the keypoints and descriptors of the train image with orb
    source_kps, source_des = orb.detectAndCompute(
        source, None
    )  #  keypoints = 500 ; descriptors = 500 with 32 integer values each

    frame_count = 0

    # Obj vertices
    vertices = np.array(obj.vertices, dtype=np.float32)
    # Texture coordinates
    texcoords = np.array(obj.texcoords, dtype=np.float32)
    # Scaling matrix
    scale_matrix = np.eye(3, dtype=np.float32) * MODEL_OBJ_SCALE
    # Train image dimensions
    source_h, source_w = source.shape[:2]

    while True:
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # read the current frame
        ok, frame = cap.read()
        if not ok:
            print("Cannot read video file")
            sys.exit()

        # find the keypoints and descriptors of the frame
        frame_kps, frame_des = orb.detectAndCompute(frame, None)
        if frame_des is None:
            continue

        # match frame descriptors with source descriptors
        matches = bf.match(source_des, frame_des)

        # sort them in the order of their distance
        # the lower the distance, the better the match
        matches = sorted(matches, key=lambda x: x.distance)

        # close
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Show the image even if no matches are found, otherwise we can't see
        # the video stream
        cv2.imshow("Frame", frame)

        # verify if enough matches are found. If yes, compute Homography
        if len(matches) > MIN_MATCHES:
            H = findHomography(source_kps, frame_kps, matches)

            # draw rectangle if there is this arg
            if args.rectangle:
                drawRectangle(source, H, frame)

            # if a valid homography was found, render the cube on model plane
            if H is not None:
                try:
                    # obtain the projection matrix
                    P = projection_matrix(K, H)
                    # project cube or model
                    frame = render(
                        frame=frame,
                        obj=obj,
                        projection=P,
                        vertices=vertices,
                        texcoords=texcoords,
                        scale_matrix=scale_matrix,
                        source_h=source_h,
                        source_w=source_w,
                    )
                except Exception as e:
                    print(e)
                    pass

            if args.matches:
                # draw the first MIN_MATCHES
                frame = cv2.drawMatches(
                    source,
                    source_kps,
                    frame,
                    frame_kps,
                    matches[:MIN_MATCHES],
                    None,
                    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                )

            # show the results
            cv2.imshow("Frame", frame)

        else:
            print("Not enough matches found - %d/%d " % (len(matches), MIN_MATCHES))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
