from socket import *

ip = '192.168.45.179'
port = 8080

sever_socket = socket(AF_INET, SOCK_STREAM)
sever_socket.bind((ip,port))
sever_socket.listen(1)

print("Waiting for Client")

client_socket, addr = sever_socket.accept()

print(str(addr), '에서 접속되었습니다.')

while True:
	sendData = input('>>>>')
	client_socket.send(sendData.encode('utf-8'))

	recvData = client_socket.recv(1024)
	print('Client : ', recvData.decode('utf-8'))