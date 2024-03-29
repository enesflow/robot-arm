from adafruit_servokit import ServoKit
from time import sleep
kit = ServoKit(channels=16)

while True:
	for i in range(-180,181):
		kit.servo[7].angle = abs(i)
		sleep(0.02)
