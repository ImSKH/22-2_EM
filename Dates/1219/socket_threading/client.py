from socket import *

ip = '192.168.45.179'
port = 8080

client_socket = socket(AF_INET, SOCK_STREAM)
client_socket.connect((ip, port))

print('Connected')

while True:
	recvData = client_socket.recv(1024)
	print('Client : ', recvData.decode('utf-8'))

	sendData = input('>>>>')
	client_socket.send(sendData.encode('utf-8'))