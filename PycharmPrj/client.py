import socket

HOST = 'localhost'
PORT = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client_socket.sendall('HeLLo'.encode())

data = client_socket.recv(1024)
print('Received', repr(data.decode()))

client_socket.close()
