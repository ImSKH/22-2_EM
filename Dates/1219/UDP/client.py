import socket

host = 	'192.168.45.179'
port = 8080

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
	data = input()
	sock.sendto(data.encode(),(host,port))