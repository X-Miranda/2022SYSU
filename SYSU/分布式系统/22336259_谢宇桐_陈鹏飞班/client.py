import xmlrpc.client
import sys
import threading
import shlex
import socket


class Client:
    def __init__(self, user_id, port=8000):
        self.user_id = user_id
        self.port = port
        self.client_proxy = xmlrpc.client.ServerProxy(f"http://localhost:{self.port}/")
        self.socket_port = port + 1  # 套接字端口
        self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_client.connect(("localhost", self.socket_port))
        print("Connected to the server via socket.")

        # 启动接收通知的线程
        threading.Thread(target=self.receive_notifications, daemon=True).start()

    def receive_notifications(self):
        while True:
            try:
                message = self.socket_client.recv(1024).decode()
                if message:
                    print(f"Notification: {message.strip()}")
                    # print(self.client_proxy.get_doc())
            except Exception as e:
                print(f"Error receiving notification: {e}")
                # break

    def edit_doc(self, content):
        try:
            self.client_proxy.edit_doc((self.user_id, content))
            print("Document after edit:")
            if(self.client_proxy.get_doc()):
                print(self.client_proxy.get_doc())
        except Exception as e:
            print(f"Error editing document: {e}")

    def delete_doc(self, text):
        try:
            self.client_proxy.delete_doc((self.user_id, text))
            print("Document after deletion:")
            print(self.client_proxy.get_doc())
        except Exception as e:
            print(f"Error deleting from document: {e}")

    def replace_doc(self, old_text, new_text):
        try:
            self.client_proxy.replace_doc((self.user_id, old_text, new_text))
            print("Document after replacement:")
            print(self.client_proxy.get_doc())
        except Exception as e:
            print(f"Error replacing in document: {e}")

    def view_doc(self):
        try:
            print("Current document content:")
            print(self.client_proxy.get_doc())
        except Exception as e:
            print(f"Error viewing document: {e}")


def print_help():
    print("Client Help:")
    print("  Commands:")
    print("    edit \"content\"   - Edit the document with the given content.")
    print("    delete \"text\"    - Delete the text from the document.")
    print("    replace \"old_text\" \"new_text\" - Replace old_text with new_text in the document.")
    print("    view            - View the current document.")
    print("    exit           - Exit the system.")
    print("    help           - Show this help message.")


def command_handler(client, command, args):
    if command == "edit":
        if len(args) > 0:
            client.edit_doc(" ".join(args))
        else:
            print("Please specify content to edit.")
    elif command == "delete":
        if len(args) > 0:
            client.delete_doc(" ".join(args))
        else:
            print("Please specify text to delete.")
    elif command == "replace":
        if len(args) >= 2:
            client.replace_doc(args[0], args[1])
        else:
            print("Please specify old text and new text to replace.")
    elif command == "view":
        client.view_doc()
    elif command == "exit":
        print(f"User {client.user_id} has logged out.")
        sys.exit(0)
    else:
        print("Invalid command or arguments.")
        print_help()


def main_loop(user_id, port=8000):
    # 初始化客户端
    client = Client(user_id, port)
    print(f"User {user_id} has logged in. Type 'help' for commands.")
    while True: # 进入主循环，等待用户输入命令
        try:
            input_cmd = input("Enter command: ").strip() # 解析用户输入的命令和参数
            if not input_cmd:
                continue
            command_parts = shlex.split(input_cmd)
            cmd = command_parts[0].lower()
            args = command_parts[1:]
            # 调用 command_handler 函数处理用户输入的命令
            if cmd in ["edit", "delete", "replace", "view", "exit", "help"]:
                command_handler(client, cmd, args)
            else:
                print("Invalid command or arguments.")
                print_help()
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    # 检查命令行参数是否正确
    if len(sys.argv) != 2 or sys.argv[1].lower() == "help":
        print("Usage: python client1.py <user_id>")
        print_help()
        sys.exit(1)
    # 否则，从命令行参数中获取用户 ID，调用 main_loop 函数启动客户端
    user_id = sys.argv[1]
    main_loop(user_id)