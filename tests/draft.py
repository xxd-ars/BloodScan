import threading
import time

def print_message(message):
    for _ in range(5):
        time.sleep(1)
        print(message)

# 创建线程
thread1 = threading.Thread(target=print_message, args=("Thread 1 is running",))
thread2 = threading.Thread(target=print_message, args=("Thread 2 is running",))

# 启动线程
thread1.start()
thread2.start()

# 等待所有线程完成
thread1.join()
thread2.join()

print("All threads completed.")