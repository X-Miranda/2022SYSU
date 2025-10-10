import grpc
from rpc.pubsub_pb2 import mes2client, mes2server
from rpc.pubsub_pb2_grpc import pubsubStub


def run():
    # 配置通信的服务器IP地址和端口
    with grpc.insecure_channel('localhost:50000') as channel:
        # 创建客户端存根
        stub = pubsubStub(channel)

        while True:  # 添加无限循环，持续接收消息
            # 向服务器发送请求，等待服务器的响应
            response = stub.pubsubServe(mes2server(mes1='client'), timeout=500)

            # 打印服务器返回的消息
            print("Server response: " + response.mes2)


if __name__ == '__main__':
    run()