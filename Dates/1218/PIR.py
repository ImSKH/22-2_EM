import RPi.GPIO as GPIO

import time



GPIO.setmode(GPIO.BCM)

GPIO.setmode(6, GPIO.OUT)


pirPin = 7

GPIO.setup(pirPin, GPIO.IN, GPIO.PUD_UP)



while True:

    if GPIO.input(pirPin) == GPIO.LOW:
        GPIO.output(6, GPIO.HIGH)
    else:
        GPIO.output(6, GPIO.LOW)
    time.sleep(0.2)