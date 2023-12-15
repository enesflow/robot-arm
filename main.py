import cv2
import mediapipe as mp
from pprint import pprint
img = cv2.imread("tu.jpg", cv2.IMREAD_COLOR)
# track the hand and save the landmarks as a list
# the print the list, then save the image with the landmarks as a jpg
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_size = 2

def convert_to_list(hand): 
	wrist = hand.landmark[0]
	fingers_indexes = {
		"pinky": (17, 19),
		"ring": (13, 16),
		"middle": (9, 12),
		"index": (5, 8),
		"thumb": (1, 4), # inclusive
	}
	fingers = {
		
	}
	for finger in fingers_indexes:
		fingers[finger] = []
		for i in range(fingers_indexes[finger][0], fingers_indexes[finger][1] + 1):
			fingers[finger].append(hand.landmark[i])
	# find the middle of the hand
	tx = wrist.x
	ty = wrist.y
	tz = wrist.z
	for finger in fingers:
		tx += fingers[finger][0].x
		ty += fingers[finger][0].y
		tz += fingers[finger][0].z
	return {
		"wrist": wrist,
		"fingers": fingers,
		"middle": {
			"x": tx / 6,
			"y": ty / 6,
			"z": tz / 6
		}
	}

def z_to_size(z):
	return min(20, max(5, abs(5-(int(z * 5 * 10 ** 1.75)))))

finger_colors = {
	"thumb": (0,0,255),
	"index": (0,255,0),
	"middle": (255,0,0),
	"ring": (255,255,0),
	"pinky": (0,255,255)
}

with mp_hands.Hands(
		static_image_mode=True,
		max_num_hands=1,
		min_detection_confidence=0.5) as hands:
		results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		annotated_image = img.copy()
		data = results.multi_hand_landmarks
		# goes from red to green
		pretty_data = convert_to_list(data[0])
		# write "Wrist" on the wrist, and put a dot on it
		cv2.putText(annotated_image, "Wrist",
			(int(pretty_data["wrist"].x * img.shape[1] + 15),
			int(pretty_data["wrist"].y * img.shape[0])),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), drawing_size // 2, cv2.LINE_AA)
		wrist_size = z_to_size(pretty_data["wrist"].z)
		cv2.circle(annotated_image,
			(int(pretty_data["wrist"].x * img.shape[1]),
			int(pretty_data["wrist"].y * img.shape[0])),
			wrist_size, (0,0,0), drawing_size)
			# write "middle" on the middle, and put a dot on it
		cv2.putText(annotated_image, "Middle",
			(int(pretty_data["middle"]["x"] * img.shape[1] + 15),
			int(pretty_data["middle"]["y"] * img.shape[0])),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), drawing_size // 2, cv2.LINE_AA)
		middle_size = z_to_size(pretty_data["middle"]["z"])
		cv2.circle(annotated_image,
			(int(pretty_data["middle"]["x"] * img.shape[1]),
			int(pretty_data["middle"]["y"] * img.shape[0])),
			middle_size, (0,0,0), drawing_size)

		# write the finger names on the fingers, and put lines between the landmarks
		for finger in pretty_data["fingers"]:
			length_finger = len(pretty_data["fingers"][finger])
			for i in range(length_finger):
				# color = finger_colors[finger]
				# make the color more vibrant as the index increases
				color = (
					int(finger_colors[finger][0] * (i + 2) / length_finger),
					int(finger_colors[finger][1] * (i + 2) / length_finger),
					int(finger_colors[finger][2] * (i + 2) / length_finger)
				)
				j = i + 1
				# also put dots on the landmarks
				size = z_to_size(pretty_data["fingers"][finger][i].z)
				cv2.circle(annotated_image,
					(int(pretty_data["fingers"][finger][i].x * img.shape[1]),
					int(pretty_data["fingers"][finger][i].y * img.shape[0])),
					# make the size of the dot, the depth of the landmark, so closer landmarks are bigger
					size, color, drawing_size)
				if (j != length_finger):
					cv2.line(annotated_image, 
					(int(pretty_data["fingers"][finger][i].x * img.shape[1]), 
					int(pretty_data["fingers"][finger][i].y * img.shape[0])), 
					(int(pretty_data["fingers"][finger][j].x * img.shape[1]), 
					int(pretty_data["fingers"][finger][j].y * img.shape[0])), 
					color, 
					drawing_size)
			cv2.putText(annotated_image, finger,
				(int(pretty_data["fingers"][finger][-1].x * img.shape[1] + 15),
				int(pretty_data["fingers"][finger][-1].y * img.shape[0])),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)


		cv2.imwrite("hand_landmarks.jpg", annotated_image)

