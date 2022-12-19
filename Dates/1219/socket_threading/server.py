##https://seolin.tistory.com/98
from socket import *
import threading 
import time


ip = '192.168.45.179'
port = 8080

def send(sock):
	while True:
		sendData = input('>>>>')
		sock.send(sendData.encode('utf-8'))

def receive(sock):
	while True:
		recvData = sock.recv(1024)
		print('Client : ', recvData.decode('utf-8'))

sever_socket = socket(AF_INET, SOCK_STREAM)
sever_socket.bind((ip,port))
sever_socket.listen(1)

print("Waiting for Client")

client_socket, addr = sever_socket.accept()

print(str(addr), '에서 접속되었습니다.')

sender = threading.Thread(target=send, args = (client_socket,))
receiver = threading.Thread(target=receive, args = (client_socket,))

sender.start()
receiver.start()

while True:
	time.sleep(1)
	pass