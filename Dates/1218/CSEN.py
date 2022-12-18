import RPi.GPIO as GPIO
import time
import I2C_LCD_driver

GPIO.setmode(GPIO.BCM)

lcd=I2C_LCD_driver.lcd()
lcd.backlight(1)

TRIG = 23
ECHO = 24

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)
time.sleep(1)

try :
	while True:
		GPIO.output(TRIG, True)
		time.sleep(0.00001)
		GPIO.output(TRIG, False)

		while GPIO.input(ECHO) == 0:
			start = time.time()

		while GPIO.input(ECHO) == 1:
			stop = time.time()

		check_time = stop - start
		dist = check_time*34300/2
		print("Distance : %.1f cm" %dist)
		time.sleep(1)

except KeyboardInterrupt:
	print("Quit Program")
	GPIO.cleanup()
