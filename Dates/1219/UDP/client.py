import socket

HOST = 	'192.168.45.179'
PORT = 8080

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_socket.sendto(b'\x02\x52\32\03', (HOST, port))