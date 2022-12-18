import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setmode(6, GPIO.OUT)
GPIO.setmode(17, GPIO.IN)

try :
	while True:
		inputIO = GPIO.input(17)

		if inputIO == False:
			GPIO.output(6, GPIO.HIGH)

		else:
			GPIO.output(6, GPIO.LOW)

except KeyboardInterrupt:
	GPIO.output(6, GPIO.LOW)

finally:
	GPIO.cleanup()