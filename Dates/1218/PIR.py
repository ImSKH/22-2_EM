import RPi.GPIO as GPIO
import I2C_LCD_driver
import time



GPIO.setmode(GPIO.BCM)

pirPin = 7

GPIO.setup(pirPin, GPIO.IN)

lcd=I2C_LCD_driver.lcd()
lcd.backlight(1)
try:
    while True:
        pir_state = GPIO.input(pirPin)
        if pir_state == True:
            print("undectected")
            lcd.lcd_display_string("NNNNN",1)
            time.sleep(0.2)
        else:
            print("dectected")
            lcd.lcd_display_string("YYYYY",1)
            time.sleep(0.2)

except KeyboardInterrupt :
    lcd.backlight(0)
    GPIO.cleanup()
