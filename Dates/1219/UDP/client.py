import socket

host = 	'192.168.45.179'
port = 8080

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
	data = input()
	client_socket.sendto(data.encode(),(host,port))

	data, address = client_socket.recvfrom(200)

	print("Client send and received data : ", data.decode())
	print("data form IP : ",address[0])
	print("data form PORT : ",address[1])