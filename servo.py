import RPi.GPIO as GPIO
import time
import cv2 as cv
import mediapipe as mp
import numpy as np
import math
from pprint import pprint
from picamera2 import Picamera2

GPIO.cleanup()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# open camera and just display
cv.startWindowThread()
picam2 = Picamera2()
w = 1920
h = 1080
z = 4
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (int(w / z), int(h / z))}))
picam2.start()

def getPWM(pin):
	GPIO.setmode(GPIO.BCM)
	GPIO.setup(pin, GPIO.OUT)

	pwm = GPIO.PWM(pin, 50) 
	pwm.start(2.5)
	return pwm
	

def setDeg(pwm, deg):
	pwm.ChangeDutyCycle(2.5 + 10 * deg / 180)

pwm = getPWM(17)


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
    "thumb": (0, 0, 255),
    "index": (0, 255, 0),
    "middle": (255, 0, 0),
    "ring": (255, 255, 0),
    "pinky": (0, 255, 255)
}

# function to draw the hand data on the frame
def draw_hand(frame, data):
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
    # cv.fillPoly(frame, np.int32([points * [frame.shape[1], frame.shape[0]]]), finger_colors["wrist"])
    # now that'd fill the palm, i just want to outline
    cv.polylines(frame, np.int32([points * [frame.shape[1], frame.shape[0]]]), True, finger_colors["wrist"], z_to_size(data["wrist"]["z"]))
    up = 0
    for finger in data["fingers"]:
        f = data["fingers"][finger]
        arr = []
        # arr.append(180 - calculate_angle(data["wrist"], f[0], f[1]))
        arr.append(180 - calculate_angle(data["wrist"], f[0], f[1]))
        arr.append(180 - calculate_angle(f[0], f[1], f[2]))
        #a = put_angle(frame, data["wrist"], f[0], f[1], finger_colors[finger])
        #b = put_angle(frame, f[0], f[1], f[2], finger_colors[finger])
        if finger != "pinky":
            pass
            #up += 0 if b < 160 else 1
            #put_angle(frame, f[1], f[2], f[3], finger_colors[finger])
            # arr.append(180 - calculate_angle(f[1], f[2], f[3]))
        avg=min(arr)
        #if avg <= 100: avg /= 5
        cv.putText(frame, f"{avg}",
            (int(f[1]["x"] * frame.shape[1] + 15),
            int(f[1]["y"] * frame.shape[0])),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, finger_colors[finger], 1, cv.LINE_AA)
        if finger == "index":
            setDeg(pwm, avg)
        up += 0 if avg < 152 else 1
    # put number "up" on the top left
    cv.putText(frame, str(up), (50, 150), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2, cv.LINE_AA)
    return frame


def calculate_angle(a, b, c):
  """
  Calculates the angle between three points in degrees, considering all three axes.

  Args:
      a: A dictionary containing the x, y, and z coordinates of point A.
      b: A dictionary containing the x, y, and z coordinates of point B.
      c: A dictionary containing the x, y, and z coordinates of point C.

  Returns:
      The angle between points A, B, and C in degrees.
  """
  div = 2
  a["z"] /= div
  b["z"] /= div
  c["z"] /= div
  ab = np.array([b["x"] - a["x"], b["y"] - a["y"], b["z"] - a["z"]])  # Vector AB
  bc = np.array([c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"]])  # Vector BC
  dot_product = np.dot(ab, bc)  # Dot product of AB and BC
  magnitudes = np.linalg.norm(ab) * np.linalg.norm(bc)  # Magnitudes of AB and BC
  angle = np.arccos(dot_product / magnitudes)  # Angle in radians
  return round(int(np.rad2deg(angle)) / 10) * 10  # Convert to degrees

def put_angle(frame, a, b, c, color=(0, 0, 0)):
    angle = round(180 - calculate_angle(a, b, c))
    # cv.putText(frame, str(angle), (int(center["x"] * frame.shape[1]), int(center["y"] * frame.shape[0])), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv.LINE_AA)
    # write text on b
    cv.putText(frame, f"{angle}",
        (int(b["x"] * frame.shape[1] + 15),
        int(b["y"] * frame.shape[0])),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
    return angle

try:
    while True:
        frame = picam2.capture_array()
        hand = get_hand(frame)
        time.sleep(0.6)
        if hand:
            hand_data = get_hand_data_formatted(hand)
            frame = draw_hand(frame, hand_data)
        cv.imshow("frame", frame)
        cv.waitKey(1)

        
except:
    print("Exiting")
    picam2.close()
    cv.destroyAllWindows()
    pwm.stop()
    GPIO.cleanup()

