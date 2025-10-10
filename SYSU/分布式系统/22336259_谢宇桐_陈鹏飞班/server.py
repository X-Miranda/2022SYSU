import xmlrpc.server
import threading
import socket


class SharedDoc:
    def __init__(self):
        self.doc = ""
        self.lock = threading.Lock()
        self.clients = {}  # 使用字典存储客户端的套接字

    def edit(self, user_id, content):
        with self.lock:
            print(f"{user_id} is editing the document.")  # 告诉服务器端用户正在操作文档
            self.doc += content  # 将 content 添加到文档末尾
            self.notify_clients(user_id, "edit", content)  # 调用notify_clients通知所有客户端文档已被编辑
            print(f"Document after edit by {user_id}: {self.doc}")  # 打印当前文档内容

    def delete(self, user_id, text):
        with self.lock:
            print(f"{user_id} is deleting text from the document.")
            self.doc = self.doc.replace(text, "")  # 使用replace方法进行删除文档中的 text
            self.notify_clients(user_id, "delete", text)
            print(f"Document after deletion by {user_id}: {self.doc}")

    def replace(self, user_id, old_text, new_text):
        with self.lock:
            print(f"{user_id} is replacing text in the document.")
            self.doc = self.doc.replace(old_text, new_text)  # 使用replace方法将文档中的old_text替换为new_text
            self.notify_clients(user_id, "replace", old_text + " -> " + new_text)
            print(f"Document after replacement by {user_id}: {self.doc}")

    def get_doc(self):
        return self.doc  # 返回当前文档的内容

    def register_client(self, client_socket):
        self.clients[client_socket.fileno()] = client_socket

    def notify_clients(self, user_id, action, details):
        message = f"{user_id} {action} '{details}'\n"  # 通知消息
        for client_socket in self.clients.values():  # 遍历所有已注册的客户端套接字
            try:
                client_socket.sendall(message.encode())  # 将通知消息发送给每个客户端
            except Exception as e:
                print(f"Error notifying client: {e}")  # 如果发送失败，打印错误信息


class DocService:
    def __init__(self, shared_doc):
        # 将SharedDocument的方法暴露给 XML-RPC 客户端
        self.shared_doc = shared_doc

    def edit_doc(self, params):
        user_id, content = params # 解析参数
        self.shared_doc.edit(user_id, content) # 调用方法
        return f"Edit by {user_id} successful."

    def delete_doc(self, params):
        user_id, text = params
        self.shared_doc.delete(user_id, text)
        return f"Deletion by {user_id} successful."

    def replace_doc(self, params):
        user_id, old_text, new_text = params
        self.shared_doc.replace(user_id, old_text, new_text)
        return f"Replacement by {user_id} successful."

    def get_doc(self):
        return self.shared_doc.get_doc()

    def register_client(self, client_socket):
        self.shared_doc.register_client(client_socket)


def start_server(shared_doc, port=8000):
    print("Attempting to start server on port 8000...")
    try:
        # 启动 XML-RPC 服务器
        server = xmlrpc.server.SimpleXMLRPCServer(("localhost", port))
        # 将 DocumentService 实例注册到服务器，使其方法可以通过 XML-RPC 调用
        server.register_instance(DocService(shared_doc))
        print("Server started on port 8000...")

        # 启动套接字服务器
        sock_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock_server.bind(("localhost", port + 1))  # 使用一个额外的端口
        sock_server.listen(5)
        print(f"Socket server started on port {port + 1}...")

        def handle_socket_connections():
            while True: # 不断接受新的客户端连接
                # 将客户端套接字注册到 SharedDoc 中
                client_socket, addr = sock_server.accept()
                print(f"New client connected: {addr}")
                shared_doc.register_client(client_socket)

        threading.Thread(target=handle_socket_connections, daemon=True).start()

        server.serve_forever()
    except socket.error as e:
        print(f"Socket error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    shared_doc = SharedDoc()
    server_thread = threading.Thread(target=start_server, args=(shared_doc, 8000))
    server_thread.start()