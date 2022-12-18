import RPi.GPIO as GPIO
import I2C_LCD_driver.py

GPIO.setmode(GPIO.BCM)
#GPIO.setup(6, GPIO.OUT)
GPIO.setup(17, GPIO.IN)

lcd = I2C_LCD_driver.lcd()
lcd.backlight(1)
try :
	while True:
		inputIO = GPIO.input(17)

		if inputIO == False:
			#GPIO.output(6, GPIO.HIGH)
			lcd.lcd_display_string("BUTTON",1)
		else:
			#GPIO.output(6, GPIO.LOW)
			lcd.lcd_display_string("NO BUTTON",1)

except KeyboardInterrupt:
	#GPIO.output(6, GPIO.LOW)
	lcd.backlight(0)
finally:
	GPIO.cleanup()