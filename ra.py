import cv2

MIN_MATCHES = 15

scene = cv2.imread('scene.jpg', 0)
model = cv2.imread('model.jpg', 0)

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