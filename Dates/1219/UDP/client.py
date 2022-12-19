import socket

HOST = '192.168.222.23'
PORT = 8080

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

while True:
	message = input('Enter message : ')
	if message == 'q':
		break

	client_socket.send(message.encode())

	data = client_socket.recv(1024)

	print('Received from the server : ', repr(data.decode()))

client_socket.close()