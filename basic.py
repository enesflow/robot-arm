import RPi.GPIO as GPIO
import time

servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)
pwm = GPIO.PWM(servoPIN, 50) 
pwm.start(2.5)
def setDeg(pwm, deg):
	pwm.ChangeDutyCycle(2.5 + 10 * deg / 180)

try:
	while True:
		setDeg(pwm, 0)
		time.sleep(1)
		setDeg(pwm, 90)
		time.sleep(1)
		setDeg(pwm, 180)
		time.sleep(1)
except:
	print("Exiting")
	pwm.stop()
	GPIO.cleanup() 
