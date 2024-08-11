import machine
import pycom


exec(open('main.py').read())
exec(open('wifi.py').read())
exec(open('data_read.py').read())
exec(open('MQTT_client.py').read())