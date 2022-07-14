from time import sleep
import socket


def main():
    # udp 通信地址，IP+端口号
    udp_addr = ('128.40.42.75', 19999)
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind(udp_addr)

    while True:
        recv_data = udp_socket.recvfrom(1024)  # 1024表示本次接收的最大字节数
        print("[From %s:%d]:%s" % (recv_data[1][0], recv_data[1][1], recv_data[0].decode("utf-8")))


if __name__ == '__main__':
    print("udp server ")
    main()