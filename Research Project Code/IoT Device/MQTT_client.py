from mqtt import MQTTClient
from network import WLAN
import machine
import time

FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60

def sub_cb(topic, msg):
   print("Message received: "+str(msg))


client = MQTTClient("device_id", "broker.hivemq.com", port=1883)
client.set_callback(sub_cb)
client.connect()
client.subscribe(topic="youraccount/DATA")  

def on_disconnect(client, userdata, rc):
    logging.info("Disconnected with result code: %s", rc)
    reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY
    while reconnect_count < MAX_RECONNECT_COUNT:
        logging.info("Reconnecting in %d seconds...", reconnect_delay)
        time.sleep(reconnect_delay)

        try:
            client.reconnect()
            logging.info("Reconnected successfully!")
            return
        except Exception as err:
            logging.error("%s. Reconnect failed. Retrying...", err)

        reconnect_delay *= RECONNECT_RATE
        reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
        reconnect_count += 1
    logging.info("Reconnect failed after %s attempts. Exiting...", reconnect_count)

client.on_disconnect = on_disconnect

while True: 
    try:
        client.publish(topic="youraccount/DATA", msg="Temperature: " + str(alt.temperature())) 
        client.publish(topic="youraccount/DATA", msg="Altitude: " + str(alt.altitude()))
        client.check_msg()
        time.sleep(30)
    except:
        on_disconnect(client, userdata, rc)