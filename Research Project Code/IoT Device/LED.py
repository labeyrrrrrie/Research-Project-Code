import pycom
import time
pycom.heartbeat(False)
print("HELLO world pchanging colors")
for cycles in range(2): #stop after 2 cycles
    print("green")
    pycom.rgbled(0x007f00) #green
    time.sleep(1)
    print("yellow")
    pycom.rgbled(0x7f7f00) #yellow
    time.sleep(1.5)
    print("red")
    pycom.rgbled(0x7f0000) #red
    time.sleep(1)