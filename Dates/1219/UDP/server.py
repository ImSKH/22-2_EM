#https://ekfkdlxm.tistory.com/31
import socket

host = '192.168.45.179'
port = 8080

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

server_socket.bind((host, port))

data, address = server_socket.recvform(200)