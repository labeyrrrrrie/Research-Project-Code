import time
import paho.mqtt.subscribe as subscribe
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from openpyxl import Workbook

array_Tem = []
array_Alt = []

def update(): 
    fig, ax = plt.subplots()
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    x = [tup[0] for tup in array_Tem]
    y = [tup[1] for tup in array_Tem]
    print(f'the values are: {y}')
    plt.xlim([x[0],x[-1]])
    plt.plot(x,y)
    plt.show()
    
def print_msg(client, userdata, message):
     #Convert to String
     payload_str = message.payload.decode('utf-8').strip()
     if('finish' in str(payload_str)):
        update()
     
     parts = payload_str.split(':')
    
     #Extract identifiers and numerical values
     identifier = parts[0].strip()
     value = parts[1].strip()
     head_identifier = identifier[0]
    
     result = f"{head_identifier}:{value}"

     value_double = float(value)

     unix = int(time.time())

     if head_identifier == 'T':
        array_Tem.append([unix, value_double])
     elif head_identifier == 'A':
        array_Alt.append([unix, value_double])

     print(result)
     print("T[]=",array_Tem)
     print("A[]=",array_Alt)

     #List of column names
     column_names = ['Datetime', 'tem_data']
     #Create a DataFrame object
     df = pd.DataFrame(array_Tem, columns=column_names)
     #CSV file path
     csv_file_path = 'D:\csvdata\matrix_tem_data.csv'
     #Write the DataFrame to a CSV file
     df.to_csv(csv_file_path, index=False)

subscribe.callback(print_msg, "youraccount/DATA", hostname="broker.hivemq.com")

ani = FuncAnimation(fig, update, interval=1000)
