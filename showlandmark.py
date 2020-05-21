
import argparse
import skimage.transform
import dlib
import cv2
import numpy as np
from scipy.spatial import Delaunay
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
import face_alignment

def face_landmarks(image, exactly_count=None):
    faces = detector(image)
    landmarks = []
    for face in faces:
        shape = predictor(image, face)
        landmarks.append(shape)
        if exactly_count is not None and len(landmarks) >= exactly_count:
            break
    if exactly_count is not None and len(landmarks) < exactly_count:
        raise RuntimeError(f'image has no exactly {exactly_count} face')
    return landmarks




def shape_to_array(sh):
    return np.array([[p.x, p.y] for p in sh.parts()], dtype=float)

target = [(125, 224), (134, 245), (139, 266), (149, 298), (161, 325), (178, 350), (198, 371), (220, 385), (247, 389), (278, 385), (303, 373), (323, 353), (340, 325), (353, 299), (361, 269), (367, 246), (378, 220), (145, 206), (169, 190), (189, 186), (215, 192), (230, 204), (269, 201), (285, 193), (313, 186), (338, 194), (352, 208), (250, 224), (250, 244), (250, 264), (250, 279), (232, 296), (243, 298), (249, 300), (257, 298), (266, 297), (167, 217), (184, 210), (205, 211), (224, 225), (205, 238), (180, 238), (274, 225), (294, 210), (316, 209), (332, 217), (323, 235), (295, 239), (210, 324), (222, 320), (237, 320), (248, 322), (259, 319), (275, 318), (289, 322), (277, 338), (262, 342), (251, 346), (238, 344), (222, 338), (218, 326), (234, 327), (248, 328), (270, 327), (283, 325), (264, 335), (250, 337), (235, 335)]

def triangle(img,landmark):
    points = []
    points.append([0,0])
    points.append([img.shape[1]//2, 0.0])
    points.append([img.shape[1], 0.0])
    points.append([img.shape[1], img.shape[0]//2])
    points.append([img.shape[1], img.shape[0]])
    points.append([img.shape[1]//2, img.shape[0]])
    points.append([0, img.shape[0]])
    for i in range(landmark.shape[0]):
            points.append([landmark[i,0],landmark[i,1]])
    tr = Delaunay(np.array(points))
    return tr.points,tr.simplices

prev_landmark = None

def avatar_ladnmark(l,r,t):
    return t+(r-l)

if __name__ == "__main__":

    avatar = cv2.imread('iAvatar.png')
    #dst_face_landmarks = face_landmarks(avatar)

    parser = argparse.ArgumentParser()
    args = parser.parse_args()



    cam = cv2.VideoCapture(0)
    frame_size = (420, 240)

    target = [[p[0]/avatar.shape[1]*frame_size[0], p[1]/avatar.shape[0]*frame_size[1]] for p in target]
    target = np.array(target)
    avatar_show = cv2.resize(avatar, dsize=frame_size)
    tpoints,tsimplies = triangle(avatar_show, target)
    while True:

        ret, frame_in = cam.read()
        frame_in = cv2.resize(frame_in, dsize=frame_size)

        landmarks = face_landmarks(frame_in)
        if len(landmarks)<1:
            continue

        landmarks = shape_to_array(landmarks[0])

        if prev_landmark is None:
            prev_landmark = landmarks
            continue

        dest_landmark = avatar_ladnmark(prev_landmark,landmarks,target)
        #dest_landmark = target

        result = avatar_show.copy()

        for i in range(dest_landmark.shape[0]):
            cv2.circle(result, (int(dest_landmark[i,0]), int(dest_landmark[i,1])), 1, (0, 255, 0), 1)

        diff = landmarks - prev_landmark
        for i in range(tsimplies.shape[0]):
            prev = None
            for k in range(tsimplies.shape[1]):
                pi = tsimplies[i, k]
                p = tpoints[pi]
                pi = pi-7
                if pi>=49 and pi<68:
                    dp = diff[pi]
                    p = (p[0]+dp[0],p[1]+dp[1])
                if prev is None:
                    prev = p
                    continue
                cv2.line(result, (int(p[0]), int(p[1])), (int(prev[0]), int(prev[1])), (0, 255, 0), thickness=1)
                prev = p


        #for t in []:
        #    prev = None
        #    for p in t:
        #        if prev is None:
        #            prev = p
        #            continue
        #        cv2.line(result,(int(p[0]),int(p[1])),(int(prev[0]),int(prev[1])),(0,255,0),thickness=1)
        #        prev = p

        cv2.imshow('webcam', result)
        ch = 0xFF & cv2.waitKey(1)
        if ch in [27, ord('q')]:
            break

    cam.release()
    cv2.destroyAllWindows()