'''### 旋转控制模块技术文档

---

#### 概述
本模块负责通过Modbus TCP协议控制旋转和夹持设备，具体实现包括与多个从机（如机械手臂和夹持器）的通信。通过发送特定的十六进制数据指令，实现设备的初始归零、旋转和移动操作，并通过线程事件机制协调旋转和拍照过程。

---

#### API接口

1. **connect_init()**
   - **功能**：初始化并连接到旋转和夹持设备。
   - **输入**：无
   - **输出**：打印连接成功或失败的信息。

2. **connect_close()**
   - **功能**：关闭与旋转和夹持设备的连接。
   - **输入**：无
   - **输出**：无

3. **G_Send_HOM()**
   - **功能**：发送归零指令到G设备，并等待反馈确认完成。
   - **输入**：无
   - **输出**：打印归零完成信息。

4. **R_Send_HOM()**
   - **功能**：发送归零指令到R设备，并等待反馈确认完成。
   - **输入**：无
   - **输出**：打印归零完成信息。

5. **G_Send_RC1()**
   - **功能**：发送RC1指令到G设备，并等待反馈确认完成。
   - **输入**：无
   - **输出**：打印RC1完成信息。

6. **G_Send_RC2()**
   - **功能**：发送RC2指令到G设备，并等待反馈确认完成。
   - **输入**：无
   - **输出**：打印RC2完成信息。

7. **R_Send_RC1()**
   - **功能**：发送RC1指令到R设备，并等待反馈确认完成。
   - **输入**：无
   - **输出**：打印RC1完成信息，并设置`photo_event`事件，等待`rotate_event`事件。

8. **R_Send_RC2()**
   - **功能**：发送RC2指令到R设备，并等待反馈确认完成。
   - **输入**：无
   - **输出**：打印RC2完成信息。

9. **R_Send_RC3()**
   - **功能**：发送RC3指令到R设备，并等待反馈确认完成。
   - **输入**：无
   - **输出**：打印RC3完成信息，并设置`photo_event`事件，等待`rotate_event`事件。

10. **R_Send_RC4()**
    - **功能**：发送RC4指令到R设备，并等待反馈确认完成。
    - **输入**：无
    - **输出**：打印RC4完成信息。

---

#### 功能原理

1. **Modbus TCP通信**
   - 使用标准的Modbus TCP协议，通过指定的IP地址和端口，与设备进行通信。每个指令通过预定义的十六进制数据发送，并接收反馈信息以确认指令执行情况。

2. **设备归零**
   - 通过发送一系列预定义的归零指令，实现设备的归零操作。每个指令发送后，通过读取反馈数据，判断归零操作是否完成。

3. **旋转和移动操作**
   - 通过发送特定的RC指令，实现设备的旋转和移动操作。每个操作步骤分为设置指令和启动指令，发送后通过读取反馈数据，确认操作完成。

4. **线程事件机制**
   - 通过`threading.Event`实现旋转和拍照过程的协调。在旋转操作完成后，通过设置和等待事件，控制流程的同步和执行顺序。

---

#### 主要指令说明

- **T2、T3、HOM**：用于设备的归零操作，通过逐步发送指令，确保设备回到初始位置。
- **RC1、RC2、RC3、RC4**：用于控制设备的不同旋转和移动操作。每个指令分为设置和启动两个步骤，通过发送指令并接收反馈确认执行结果。

---

通过上述API接口和功能原理，本模块实现了对旋转和夹持设备的精确控制，确保设备在自动化流程中的可靠性和稳定性。
'''

import socket
import time
import threading

rotate_event = threading.Event()
photo_event = threading.Event()

# 从机IP地址和端口
SERVER_IP_G = '192.168.178.1'
SERVER_IP_R = '192.168.178.2'
SERVER_PORT = 502

# 构造Modbus TCP请求的十六进制数据
# T2 = b"\x00\x00\x00\x00\x00\x13\x00\x17\x00\x00\x00\x04\x00\x00\x00\x04\x08\x01\x00\x00\x00\x00\x00\x00\x00"
# T3 = b"\x00\x01\x00\x00\x00\x13\x00\x17\x00\x00\x00\x04\x00\x00\x00\x04\x08\x03\x00\x00\x00\x00\x00\x00\x00"
# HOM = b"\x00\x02\x00\x00\x00\x13\x00\x17\x00\x00\x00\x04\x00\x00\x00\x04\x08\x03\x05\x00\x00\x00\x00\x00\x00"
# RC4 = b"\x00\x03\x00\x00\x00\x13\x00\x17\x00\x00\x00\x04\x00\x00\x00\x04\x08\x03\x01\x04\x00\x00\x00\x00\x00"
# RC4_start = b"\x00\x03\x00\x00\x00\x13\x00\x17\x00\x00\x00\x04\x00\x00\x00\x04\x08\x03\x03\x04\x00\x00\x00\x00\x00"

T2 = b"\x00:\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x01\x00\x03\x00\x00\x03\x00\x00"
T3 = b"\x00<\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x00\x03\x00\x00\x03\x00\x00"
Rd = b"\x00;\x00\x00\x00\x06\x00\x03\x00\x00\x00\x04"
HOM = b"\x00>\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x05\x00\x00\x00\x03\x00\x00"

#G:4mm R:0.5r
RC1 = b"\x00@\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x01\x01\x00\x00\x03\x00\x00"
RC1_start = b"\x00B\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x03\x01\x00\x00\x03\x00\x00"
#G:-4mm R:-0.5r
RC2 = b"\x00@\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x01\x02\x00\x00\x03\x00\x00"
RC2_start = b"\x00B\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x03\x02\x00\x00\x03\x00\x00"
#G:2mm R:
RC3 = b"\x00@\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x01\x03\x00\x00\x03\x00\x00"
RC3_start = b"\x00B\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x03\x03\x00\x00\x03\x00\x00"
#G:-2mm R:
RC4 = b"\x00@\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x01\x04\x00\x00\x03\x00\x00"
RC4_start = b"\x00B\x00\x00\x00\x0f\x00\x10\x00\x00\x00\x04\x08\x03\x03\x04\x00\x00\x03\x00\x00"

sock_G = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
sock_R = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 


def connect_init():
    try:
        # sock_G = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        sock_G.connect((SERVER_IP_G, SERVER_PORT))
        # sock_R = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        sock_R.connect((SERVER_IP_R, SERVER_PORT))
    except Exception as e:
        print("发生异常:", e)

def connect_close():
    sock_G.close()
    sock_R.close()

def G_Send_HOM():
    sock_G.sendall(T2)
    response_hex = sock_G.recv(1024)
    while True:
        sock_G.sendall(Rd)
        response_hex = sock_G.recv(1024)
        if len(response_hex) >= 11 and response_hex[9] == 0x11:
            break
        time.sleep(0.1)
    sock_G.sendall(T3)
    response_hex = sock_G.recv(1024)
    while True:
        sock_G.sendall(Rd)
        response_hex = sock_G.recv(1024)
        if len(response_hex) >= 11 and response_hex[9] == 0x13:
            break
        time.sleep(0.1)
    sock_G.sendall(HOM)
    response_hex = sock_G.recv(1024)
    while True:
        sock_G.sendall(Rd)
        response_hex = sock_G.recv(1024)
        if len(response_hex) >= 11 and response_hex[10] == 0x87:
            break
        time.sleep(0.1)
    print("G_HOM")

def R_Send_HOM():
    sock_R.sendall(T2)
    response_hex = sock_R.recv(1024)
    while True:
        sock_R.sendall(Rd)
        response_hex = sock_R.recv(1024)
        # print(response_hex)
        if len(response_hex) >= 11 and response_hex[9] == 0x11:
            # print(response_hex)
            break
        time.sleep(0.1)
    sock_R.sendall(T3)
    response_hex = sock_R.recv(1024)
    while True:
        sock_R.sendall(Rd)
        response_hex = sock_R.recv(1024)
        if len(response_hex) >= 11 and response_hex[9] == 0x13:
            break
        time.sleep(0.1)
    sock_R.sendall(HOM)
    response_hex = sock_R.recv(1024)
    while True:
        sock_R.sendall(Rd)
        response_hex = sock_R.recv(1024)
        if len(response_hex) >= 11 and response_hex[10] == 0x87:
            break
        time.sleep(0.1)
    print("R_HOM")

def G_Send_RC1():
    sock_G.sendall(RC1)
    response_hex = sock_G.recv(1024)
    while True:
        sock_G.sendall(Rd)
        response_hex = sock_G.recv(1024)
        if len(response_hex) >= 11 and (response_hex[10] & 0x0F)== 0x05:
            break
        time.sleep(0.1)
    sock_G.sendall(RC1_start)
    response_hex = sock_G.recv(1024)
    while True:
        sock_G.sendall(Rd)
        response_hex = sock_G.recv(1024)
        if len(response_hex) >= 11 and (response_hex[10] & 0x0F) == 0x07:
            break
        time.sleep(0.1)
    print("G_RC1")

def G_Send_RC2():
    sock_G.sendall(RC2)
    response_hex = sock_G.recv(1024)
    while True:
        sock_G.sendall(Rd)
        response_hex = sock_G.recv(1024)
        if len(response_hex) >= 11 and (response_hex[10] & 0x0F)== 0x05:
            break
        time.sleep(0.1)
    sock_G.sendall(RC2_start)
    response_hex = sock_G.recv(1024)
    while True:
        sock_G.sendall(Rd)
        response_hex = sock_G.recv(1024)
        if len(response_hex) >= 11 and (response_hex[10] & 0x0F) == 0x07:
            break
        time.sleep(0.1)
    print("G_RC2")

def R_Send_RC1():
    sock_R.sendall(RC1)
    response_hex = sock_R.recv(1024)
    while True:
        sock_R.sendall(Rd)
        response_hex = sock_R.recv(1024)
        # print(response_hex)
        if len(response_hex) >= 11 and (response_hex[10] & 0x0F)== 0x05:
            # print(response_hex)
            break
        time.sleep(0.1)
    print("R_R1")
    sock_R.sendall(RC1_start)
    response_hex = sock_R.recv(1024)
    while True:
        sock_R.sendall(Rd)
        response_hex = sock_R.recv(1024)
        # print(response_hex)
        if len(response_hex) >= 11 and  (response_hex[10] & 0x0F) == 0x07:
            # print(response_hex)
            break
        time.sleep(0.1)
    print("R_RC1")
    photo_event.set()
    rotate_event.wait()
    rotate_event.clear()

def R_Send_RC2():
    sock_R.sendall(RC2)
    response_hex = sock_R.recv(1024)
    while True:
        # sock_R.sendall(RC2)
        # response_hex = sock_R.recv(1024)
        sock_R.sendall(Rd)
        response_hex = sock_R.recv(1024)
        if len(response_hex) >= 11 and response_hex[10] == 0x85:
            print(response_hex)
            break
        time.sleep(0.1)
    print("R_R2")
    sock_R.sendall(RC2_start)
    response_hex = sock_R.recv(1024)
    while True:
        # sock_R.sendall(RC2_start)
        # response_hex = sock_R.recv(1024)
        sock_R.sendall(Rd)
        response_hex = sock_R.recv(1024)
        if len(response_hex) >= 11 and response_hex[10] == 0x87:
            # print(response_hex)
            break
        time.sleep(0.1)
    print("R_RC2")

# connect_init()
# G_Send_HOM()
# R_Send_HOM()
# time.sleep(10)
# G_Send_RC1()
# G_Send_RC2()
# connect_close()
# R_Send_HOM()
# R_Send_RC1()
# R_Send_RC2()
# connect_close()
