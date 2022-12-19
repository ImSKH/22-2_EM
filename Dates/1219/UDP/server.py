#https://ekfkdlxm.tistory.com/31
import socket

host = '192.168.45.179'
port = 8080

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_socket.bind((host, port))

while True:
	data, address = server_socket.recvfrom(200)
	server_socket.sendto(data, address)
	
	print("Server received : ", data.decode())
	print("Send Client IP : ", address[0])
	print("Send Client Port : ", address[1])
	print("Server return data to ", address[0])