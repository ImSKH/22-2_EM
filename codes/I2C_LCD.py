import I2C_LCD_driver
from time import *
mylcd = I2C_LCD_driver.lcd()
mylcd.lcd_display_string("Hello World",1)
mylcd.lcd_display_string("Embeded System",2)