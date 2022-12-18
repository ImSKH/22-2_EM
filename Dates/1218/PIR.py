import RPi.GPIO as GPIO
import I2C_LCD_driver
import time



GPIO.setmode(GPIO.BCM)

pirPin = 7

GPIO.setup(pirPin, GPIO.IN, GPIO.PUD_UP)

lcd=I2C_LCD_driver.lcd()

while True:

    if GPIO.input(pirPin) == GPIO.LOW:
        lcd.lcd_display_string("YYYYY",1)
    else:
        lcd.lcd_display_string("NNNNN",1)
    time.sleep(0.2)