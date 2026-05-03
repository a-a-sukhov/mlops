import socket
import time

print("Hello from ClearML Agent")
print("hostname:", socket.gethostname())

for i in range(5):
    print("step", i)
    time.sleep(1)
