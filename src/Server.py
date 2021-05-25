import socket
import os
from _thread import *
import threading
import pickle
import numpy as np

ServerSocket = socket.socket()
host = '127.0.0.1'
port = 1233
ThreadCount = 0
try:
    ServerSocket.bind((host, port))
except socket.error as e:
    print(str(e))

print('Waitiing for a Connection..')
ServerSocket.listen(50000)
weightarray={}

def threaded_client(connection):
    #connection.send(str.encode('Welcome to the Servern'))
    while True:
        received_data=b''

        while received_data[-12:] != b'endingpickle':
            data = connection.recv(1024)
            received_data += data
        received_data = pickle.loads(received_data[:-12])
        weightarray[received_data[-1]]=received_data[:-1]
        print("receiving data from "+received_data[-1])
        #print(weightarray)
        if len(weightarray)>0:
            temparry=[]
            for val in weightarray.values():
                temparry.append(val)
            if len(temparry)>1:

                reply=list(np.mean(np.asarray(temparry),axis=0))
                print("\nWeights from clients")
                print([str(temparry[i][0][0][0])+'  ' for i in range(len(temparry))])

            else:
                reply=temparry[0]
                #print(reply)
        #print(reply)

        reply=pickle.dumps(reply)+b'endingpickle'

        connection.sendall(reply)
    connection.close()

while True:
    Client, address = ServerSocket.accept()
    print('Connected to: ' + address[0] + ':' + str(address[1]))
    start_new_thread(threaded_client, (Client, ))
    ThreadCount += 1
    print('Thread Number: ' + str(ThreadCount))
ServerSocket.close()