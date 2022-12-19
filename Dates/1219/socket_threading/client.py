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

client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect((ip, port))

print('Connected')

sender = threading.Thread(target=send, args=(client_socket,))
receiver = threading.Thread(target=receive, args=(client_socket,))

sender.start()
receiver.start()

while True:
	time.sleep(1)
	pass