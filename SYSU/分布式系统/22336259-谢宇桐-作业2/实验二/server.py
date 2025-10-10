import threading
from concurrent import futures
import grpc
from rpc.pubsub_pb2_grpc import add_pubsubServicer_to_server, pubsubServicer
from rpc.pubsub_pb2 import mes2client, mes2server
import time

# 定义服务器类
class PubsubServer(pubsubServicer):
    def __init__(self):
        self.threadLock = threading.Lock()
        self.n = 0
        self.mes = "default"
        self.mes_timestamp = 0  # 记录消息的时间戳
        self.ttl = 10  # 设置消息的TTL为10秒

    # 实现proto文件中定义的服务方法
    def pubsubServe(self, request, context):
        # 如果没有消息，则等待消息输入
        if self.n == 0:
            with self.threadLock:
                self.n += 1
                self.mes = input('mes:')
                self.mes_timestamp = time.time()  # 更新消息的时间戳
        with self.threadLock:
            current_time = time.time()
            if current_time - self.mes_timestamp > self.ttl:
                self.mes = "Message has expired"
                self.mes_timestamp = current_time
            self.n = 0
        return mes2client(mes2=self.mes)

# 创建服务器实例
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    # 将服务添加到服务器
    add_pubsubServicer_to_server(PubsubServer(), server)
    # 设置服务器监听的端口
    server.add_insecure_port('[::]:50000')
    # 启动服务器
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
