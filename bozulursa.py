
import time
import cv2 as cv
import mediapipe as mp
import numpy as np
import math
from pprint import pprint
from picamera2 import Picamera2
from adafruit_servokit import ServoKit
from pprint import pprint


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# open camera and just display
cv.startWindowThread()
picam2 = Picamera2()
w = 1920
h = 1080
z = 3
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (int(w / z), int(h / z))}))
picam2.start()
kit = ServoKit(channels=16)
servos = {
    "thumb": 0, "index": 7, "middle": 2, "ring": 3, "pinky": 4
}
maxx = 150
min_maxes = {
    "pinky": [[130, 30],[180,180]], # min, max
    "ring": [[143, 43],[180,180]], # min, max
    "middle": [[130, 50],[180,180]], # min, max
    "index": [[145, 45],[180,180]], # min, max
    "thumb": [[110, 102],[177,165]], # min, max
}

history = []


# functions for hand tracking
def get_hand(frame):
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
        results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        else:
            return None


# function to get the hand data in a nice format
def get_hand_data_formatted(hand):
    wrist = hand.landmark[0]
    # convert wrist to a dict
    wrist = {
        "x": wrist.x,
        "y": wrist.y,
        "z": wrist.z
    }
    fingers_indexes = {
        "pinky": (17, 19),
        "ring": (13, 16),
        "middle": (9, 12),
        "index": (5, 8),
        "thumb": (1, 4),  # inclusive
    }
    fingers = {

    }
    for finger in fingers_indexes:
        fingers[finger] = []
        for i in range(fingers_indexes[finger][0], fingers_indexes[finger][1] + 1):
            point = hand.landmark[i]
            fingers[finger].append({
                "x": point.x,
                "y": point.y,
                "z": point.z
            })
    # find the middle of the hand
    tx = wrist["x"]
    ty = wrist["y"]
    tz = wrist["z"]
    for finger in fingers:
        tx += fingers[finger][0]["x"]
        ty += fingers[finger][0]["y"]
        tz += fingers[finger][0]["z"]
    return {
        "wrist": wrist,
        "fingers": fingers,
        "center": {
            "x": tx / 6,
            "y": ty / 6,
            "z": tz / 6
        }
    }


def z_to_size(z):
    return min(2, max(2, abs(2 - (int(z * 5 * 10 ** 1.75)))))

def draw_dot(frame, x, y, z, color, size):
    cv.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), size, color, -1)
def draw_line(frame, x1, y1, z1, x2, y2, z2, color, size):
    cv.line(frame,
            (int(x1 * frame.shape[1]),
             int(y1 * frame.shape[0])),
            (int(x2 * frame.shape[1]),
             int(y2 * frame.shape[0])),
            color, size)


finger_colors = {
    "wrist": (255, 255, 255),
    "center": (255, 255, 255),
    "center": (255, 255, 255),
    "thumb": (0, 0, 255),
    "index": (0, 255, 0),
    "middle": (255, 0, 0),
    "ring": (255, 255, 0),
    "pinky": (0, 255, 255)
}

def scale_number(unscaled, to_min, to_max, from_min, from_max):
    unscaled -= from_min
    from_max -= from_min
    from_min = 0
    if from_max == 0:
        from_max = 1
    return maxx - ((to_max-to_min)*(unscaled-from_min)/(from_max-from_min)+to_min)
def calculate_avg(wrist, f, finger):
    return calculate_angle(wrist,f[0],f[-2])
    #arr = []
    #wrist_to_f1 = calculate_angle(wrist, f[0], f[1])
    #f1_to_f2 = calculate_angle(f[0], f[1], f[2])
    #if finger == "thumb":
    #    wrist_to_f1 = calculate_angle(f[0], f[1], f[2])
    #    f1_to_f2 = calculate_angle(f[1], f[2], f[3])
    #    if wrist_to_f1 < min_maxes[finger][0][0] + 10:
    #        f1_to_f2 = min_maxes[finger][0][1]
    #    elif f1_to_f2 < min_maxes[finger][0][1] + 10:
    #        wrist_to_f1 = min_maxes[finger][0][0]
    #arr.append(scale_number(wrist_to_f1, maxx, 0, min_maxes[finger][0][0],min_maxes[finger][1][0]))
    #arr.append(scale_number(f1_to_f2, maxx, 0, min_maxes[finger][0][1],min_maxes[finger][1][1]))
    #return sorted([0,round(sum(arr) / len(arr)),maxx])[1]

# function to draw the hand data on the frame
cnt = 0
ctime = 18
avg = {}
prev = {}
frmcnt = 3
def draw_hand(frame, data):
    global cnt
    # draw dots first
    for finger in data["fingers"]:
        for point in data["fingers"][finger]:
            draw_dot(frame, point["x"], point["y"], point["z"], finger_colors[finger], z_to_size(point["z"]))
    draw_dot(frame, data["wrist"]["x"], data["wrist"]["y"], data["wrist"]["z"], finger_colors["wrist"], z_to_size(data["wrist"]["z"]))
    draw_dot(frame, data["center"]["x"], data["center"]["y"], data["center"]["z"], finger_colors["center"], z_to_size(data["center"]["z"]))
    # draw lines
    for finger in data["fingers"]:
        for i in range(len(data["fingers"][finger]) - 1):
            draw_line(frame, data["fingers"][finger][i]["x"], data["fingers"][finger][i]["y"], data["fingers"][finger][i]["z"], data["fingers"][finger][i + 1]["x"], data["fingers"][finger][i + 1]["y"], data["fingers"][finger][i + 1]["z"], finger_colors[finger], z_to_size(data["fingers"][finger][i]["z"]))
    # connect wrist and all fingers
    for finger in data["fingers"]:
        draw_line(frame, data["wrist"]["x"], data["wrist"]["y"], data["center"]["z"], data["fingers"][finger][0]["x"], data["fingers"][finger][0]["y"], data["fingers"][finger][0]["z"], finger_colors[finger], z_to_size(data["fingers"][finger][0]["z"]))
    # draw a polygon around all finger bases and the wrist, this is the palm
    points = []
    for finger in data["fingers"]:
        points.append((data["fingers"][finger][0]["x"], data["fingers"][finger][0]["y"]))
    points.append((data["wrist"]["x"], data["wrist"]["y"]))
    points = np.array(points)
    cv.polylines(frame, np.int32([points * [frame.shape[1], frame.shape[0]]]), True, finger_colors["wrist"], z_to_size(data["wrist"]["z"]))
    p = {}
    for finger in data["fingers"]:
        f = data["fingers"][finger]
        angle = calculate_avg(data["wrist"], f, finger)
        cv.putText(frame, f"{angle}",
            (int(f[1]["x"] * frame.shape[1] + 15),
            int(f[1]["y"] * frame.shape[0])),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, finger_colors[finger], 1, cv.LINE_AA)
        #if finger == "thumb":
        #    print(calculate_angle(data["wrist"], f[0], f[1]), calculate_angle(f[0], f[1], f[2]),calculate_angle(f[1],f[2],f[3]))
        if cnt >= ctime * 2:
            if not finger in avg:
                avg[finger] = []
            avg[finger].append(angle)
            if cnt % frmcnt == 0:
                avg_angle = sum(avg[finger]) / frmcnt
                if not finger in prev or abs(avg_angle - prev[finger]) > 10:
                    kit.servo[servos[finger]].angle = 180 - avg_angle if finger != "thumb" else avg_angle
                    prev[finger] = avg_angle
                avg[finger] = []
            
            
        if finger != "thumb":
            p[finger] = [calculate_angle(data["wrist"], f[0], f[1]),calculate_angle(f[0], f[1], f[2])]
        else:
            wrist_to_f1 = calculate_angle(f[0], f[1], f[2])
            f1_to_f2 = calculate_angle(f[1], f[2], f[3])
            p[finger] = [wrist_to_f1,f1_to_f2]
    if cnt < ctime * 2:
        if cnt < ctime:
            print(f"Avucunuzu acinn {ctime - cnt}")
            cv.putText(frame, f"Avucunuzu acinn {ctime - cnt}", (50,50),
            cv.FONT_HERSHEY_SIMPLEX, 1, finger_colors[finger], 2, cv.LINE_AA)
        else:
            print(f"Avucunuzu kapayinn {ctime*2 - cnt}")
            cv.putText(frame, f"Avucunuzu kapayinn {ctime*2 - cnt}", (50,50),
            cv.FONT_HERSHEY_SIMPLEX, 1, finger_colors[finger], 2, cv.LINE_AA)
        history.append(p)
        save_res()
    cnt+=1
              
    return frame


def calculate_angle(a, b, c):
  div = 1
  a["z"] /= div
  b["z"] /= div
  c["z"] /= div
  ab = np.array([b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"]])  # Vector AB
  bc = np.array([c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"]])  # Vector BC
  dot_product = np.dot(ab, bc)  # Dot product of AB and BC
  magnitudes = np.linalg.norm(ab) * np.linalg.norm(bc)  # Magnitudes of AB and BC
  angle = np.arccos(dot_product / magnitudes)  # Angle in radians
  return 180 - round(int(np.rad2deg(angle)))  # Convert to degrees

def save_res():
    global history
    hmax = 1000
    if len(history) > hmax:
        history = history[-hmax:]
    l = min(10,len(history))
    for finger in min_maxes:
        for i in range(2):
            hs = sorted(history, key=lambda d: d[finger][i])
            t = 0
            for j in range(l):
                t += hs[j][finger][i]
            t /= l
            t2 = 0
            hs = hs[::-1]
            for j in range(l):
                t2 += hs[j][finger][i]
            t2 /= l
            t = int(t)
            t2 = int(t2)
            if t2 == t:
                t2+=1
                t-=1
            min_maxes[finger][0][i] = t
            min_maxes[finger][1][i] = t2
    #pprint(min_maxes)
    #print("--")
            
    

def put_angle(frame, a, b, c, color=(0, 0, 0)):
    angle = round(calculate_angle(a, b, c))
    # cv.putText(frame, str(angle), (int(center["x"] * frame.shape[1]), int(center["y"] * frame.shape[0])), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv.LINE_AA)
    # write text on b
    cv.putText(frame, f"{angle}",
        (int(b["x"] * frame.shape[1] + 15),
        int(b["y"] * frame.shape[0])),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    return angle

while True:
    frame = picam2.capture_array()
    hand = get_hand(frame)
    if hand:
        hand_data = get_hand_data_formatted(hand)
        frame = draw_hand(frame, hand_data)
    else:
        print("el yok!!!!")
    cv.imshow("frame", frame)
    cv.waitKey(1)

