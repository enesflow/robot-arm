import cv2 as cv
import mediapipe as mp
from picamera2 import Picamera2
import numpy as np
import math
from pprint import pprint
from adafruit_servokit import ServoKit
from pprint import pprint


mp_hands = mp.solutions.hands
picam2 = Picamera2()

# 1 ─ configure camera in 3-channel RGB to avoid later conversions
w = 1920
h = 1080
z = 3.2
picam2.configure(
    picam2.create_preview_configuration(main={"format": 'RGB888', "size": (int(w / z), int(h / z))}))
picam2.start()

# 2 ─ create the Hands object ONCE, outside the loop
hands = mp_hands.Hands(
    static_image_mode=False,      # video-stream mode
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

kit = ServoKit(channels=16)
servos = {
    "thumb": 8, "index": 7, "middle": 2, "ring": 3, "pinky": 4
}
maxx = 150
min_maxes = {
    "pinky": [0, 180],  # min, max
    "ring": [0, 180],  # min, max
    "middle": [0, 180],  # min, max
    "index": [0, 180],  # min, max
    "thumb": [0, 180],  # min, max
}

history = []
CALIB_WINDOW_SIZE = 500

# only commit servo moves when change ≥ this value
ANGLE_THRESHOLD = 15
last_angles = {}


def update_min_maxes():
    for finger in min_maxes:
        values = [h[finger] for h in history]
        if values:
            min_maxes[finger] = [min(values), max(values)]


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
    cv.circle(
        frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), size, color, -1)


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


def calculate_avg(wrist, f, finger, raw=False):
    OldValue = calculate_angle(wrist, f[0], f[-1])
    if raw:
        return OldValue
    OldMin = min_maxes[finger][0]
    OldMax = min_maxes[finger][1]
    OldRange = (OldMax - OldMin)
    if OldRange == 0:
        return OldValue
    NewMin = 0
    NewMax = 180
    NewRange = (NewMax - NewMin)
    NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

    def reversea(x):
        return (math.exp((x + 11.603) / 62.91) + 1.202) / 0.123
    return sorted([0, round(reversea(NewValue)), 180])[1]
    # arr = []
    # wrist_to_f1 = calculate_angle(wrist, f[0], f[1])
    # f1_to_f2 = calculate_angle(f[0], f[1], f[2])
    # if finger == "thumb":
    #    wrist_to_f1 = calculate_angle(f[0], f[1], f[2])
    #    f1_to_f2 = calculate_angle(f[1], f[2], f[3])
    #    if wrist_to_f1 < min_maxes[finger][0][0] + 10:
    #        f1_to_f2 = min_maxes[finger][0][1]
    #    elif f1_to_f2 < min_maxes[finger][0][1] + 10:
    #        wrist_to_f1 = min_maxes[finger][0][0]
    # arr.append(scale_number(wrist_to_f1, maxx, 0, min_maxes[finger][0][0],min_maxes[finger][1][0]))
    # arr.append(scale_number(f1_to_f2, maxx, 0, min_maxes[finger][0][1],min_maxes[finger][1][1]))
    # return sorted([0,round(sum(arr) / len(arr)),maxx])[1]


# function to draw the hand data on the frame
cnt = 0
ctime = 13
avg = {}
prev = {}
frmcnt = 3


def draw_hand(frame, data):
    # draw dots first
    for finger in data["fingers"]:
        for point in data["fingers"][finger]:
            draw_dot(frame, point["x"], point["y"], point["z"],
                     finger_colors[finger], z_to_size(point["z"]))
    draw_dot(frame, data["wrist"]["x"], data["wrist"]["y"], data["wrist"]
             ["z"], finger_colors["wrist"], z_to_size(data["wrist"]["z"]))
    draw_dot(frame, data["center"]["x"], data["center"]["y"], data["center"]
             ["z"], finger_colors["center"], z_to_size(data["center"]["z"]))
    # draw lines
    for finger in data["fingers"]:
        for i in range(len(data["fingers"][finger]) - 1):
            draw_line(frame, data["fingers"][finger][i]["x"], data["fingers"][finger][i]["y"], data["fingers"][finger][i]["z"], data["fingers"][finger][i + 1]
                      ["x"], data["fingers"][finger][i + 1]["y"], data["fingers"][finger][i + 1]["z"], finger_colors[finger], z_to_size(data["fingers"][finger][i]["z"]))
    # connect wrist and all fingers
    for finger in data["fingers"]:
        draw_line(frame, data["wrist"]["x"], data["wrist"]["y"], data["center"]["z"], data["fingers"][finger][0]["x"], data["fingers"]
                  [finger][0]["y"], data["fingers"][finger][0]["z"], finger_colors[finger], z_to_size(data["fingers"][finger][0]["z"]))
    # draw a polygon around all finger bases and the wrist, this is the palm
    points = []
    for finger in data["fingers"]:
        points.append((data["fingers"][finger][0]["x"],
                      data["fingers"][finger][0]["y"]))
    points.append((data["wrist"]["x"], data["wrist"]["y"]))
    points = np.array(points)
    cv.polylines(frame, np.int32([points * [frame.shape[1], frame.shape[0]]]),
                 True, finger_colors["wrist"], z_to_size(data["wrist"]["z"]))
    p = {}
    for finger in data["fingers"]:
        f = data["fingers"][finger]
        # raw angle
        p[finger] = calculate_angle(data["wrist"], f[0], f[-1])

    # live sliding‐window calibration
    history.append(p)
    if len(history) > CALIB_WINDOW_SIZE:
        history.pop(0)
    update_min_maxes()

    # map & drive servos every frame
    for finger, raw_ang in p.items():
        mapped = calculate_avg(data["wrist"], data["fingers"][finger], finger)
        # use mapped for thumb, inverse for others
        angle = mapped if finger == "thumb" else 180 - mapped
        # only send servo command if change exceeds threshold
        if finger not in last_angles or abs(angle - last_angles[finger]) >= ANGLE_THRESHOLD:
            kit.servo[servos[finger]].angle = angle
            last_angles[finger] = angle

    return frame


def calculate_angle(a, b, c):
    div = 1
    a["z"] /= div
    b["z"] /= div
    c["z"] /= div
    ab = np.array([b["x"] - a["x"], b["y"] - a["y"],
                  b["z"] - a["z"]])  # Vector AB
    bc = np.array([c["x"] - b["x"], c["y"] - b["y"],
                  c["z"] - b["z"]])  # Vector BC
    dot_product = np.dot(ab, bc)  # Dot product of AB and BC
    # Magnitudes of AB and BC
    magnitudes = np.linalg.norm(ab) * np.linalg.norm(bc)
    angle = np.arccos(dot_product / magnitudes)  # Angle in radians
    return 180 - round(int(np.rad2deg(angle)))  # Convert to degrees


def save_res():
    global history
    hmax = 1000
    if len(history) > hmax:
        history = history[-hmax:]
    l = min(10, len(history))
    for finger in min_maxes:
        hs = sorted(history, key=lambda d: d[finger])
        t = 0
        for j in range(l):
            t += hs[j][finger]
        t /= l
        t2 = 0
        hs = hs[::-1]
        for j in range(l):
            t2 += hs[j][finger]
        t2 /= l
        t = int(t)
        t2 = int(t2)
        if t2 == t:
            t2 += 1
            t -= 1
        min_maxes[finger][0] = t
        min_maxes[finger][1] = t2
    # pprint(min_maxes)
    # print("--")


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
    frame = picam2.capture_array()          # RGB already
    results = hands.process(frame)          # no extra colour conversion

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        hand_data = get_hand_data_formatted(hand)
        frame = draw_hand(frame, hand_data)
    else:
        print("el yok!!!!")

   # cv.imshow("frame", frame)
   # if cv.waitKey(1) & 0xFF == 27:
   #     break
while True:
    frame = picam2.capture_array()          # RGB already
    results = hands.process(frame)          # no extra colour conversion

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        hand_data = get_hand_data_formatted(hand)
        frame = draw_hand(frame, hand_data)
    else:
        print("el yok!!!!")

   # cv.imshow("frame", frame)
   # if cv.waitKey(1) & 0xFF == 27:
   #     break
