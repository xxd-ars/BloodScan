'''概述
本模块负责采血管的精确移动和定位,具体实现包括对XYZ轴和夹持旋转电机的控制。通过串口通信实现与步进电机和伺服电机的指令发送和反馈处理。

API接口
Find_serial()

    功能：查找可用的串口设备。
    输入：无
    输出：打印可用的串口设备列表。

serial_init()

    功能：初始化并打开串口通信。
    输入：无
    输出：打印串口打开成功或失败的信息。

serial_close()

    功能：关闭串口通信。
    输入：无
    输出：打印串口关闭的信息。

X_send_command_with_wait(commands, port='COM3', baudrate=115200, wait_for_1percent=True)

    功能：发送X轴的控制指令，并在需要时等待"1%"的反馈。
    输入：
    commands：包含指令字符串的列表。
    port：串口端口号（默认COM3）。
    baudrate：波特率（默认115200）。
    wait_for_1percent：是否等待"1%"的反馈（默认True）。
    输出：打印发送的指令和接收到的反馈信息。

Y_send_command_with_wait(commands, port='COM3', baudrate=115200, wait_for_2percent=True)

    功能：发送Y轴的控制指令，并在需要时等待"2%"的反馈。
    输入：
    commands：包含指令字符串的列表。
    port：串口端口号（默认COM3）。
    baudrate：波特率（默认115200）。
    wait_for_2percent：是否等待"2%"的反馈（默认True）。
    输出：打印发送的指令和接收到的反馈信息。

Z_send_command_with_wait(commands, port='COM3', baudrate=115200, wait_for_3percent=True)

    功能：发送Z轴的控制指令，并在需要时等待"3%"的反馈。
    输入：
    commands：包含指令字符串的列表。
    port：串口端口号（默认COM3）。
    baudrate：波特率（默认115200）。
    wait_for_3percent：是否等待"3%"的反馈（默认True）。
    输出：打印发送的指令和接收到的反馈信息。

replace_DI_data(commands, DI_data)

    功能：替换指令列表中的DI数据。
    输入：
    commands：包含指令字符串的列表。
    DI_data：新的DI数据值。
    输出：返回替换DI数据后的指令列表。

test_replace_DI_data(distance)

    功能：测试替换Z轴指令中的DI数据。
    输入：
    distance：Z轴移动距离。
    输出：打印替换后的Z轴指令列表。
    
Send_FindHOM()

    功能：发送XYZ轴的归零指令。
    输入：无
    输出：调用Z_send_command_with_wait、X_send_command_with_wait和Y_send_command_with_wait发送各轴归零指令。

X_Move(distance)

    功能：移动X轴指定距离。
    输入：
    distance：X轴移动距离（正负值代表方向）。
    输出：调用replace_DI_data替换DI数据，并调用X_send_command_with_wait发送指令。

Y_Move(distance)

    功能：移动Y轴指定距离。
    输入：
    distance：Y轴移动距离（正负值代表方向）。
    输出：调用replace_DI_data替换DI数据，并调用Y_send_command_with_wait发送指令。

Z_Move(distance)

    功能：移动Z轴指定距离。
    输入：
    distance：Z轴移动距离（正负值代表方向）。
    输出：调用replace_DI_data替换DI数据，并调用Z_send_command_with_wait发送指令。

    
功能原理

串口通信

    串口初始化配置波特率、数据位、停止位和校验位，通过串口发送和接收指令与电机控制器进行通信。 
归零操作

    归零操作通过发送一系列预设指令实现电机的归零，确保电机在初始位置。
移动控制

    通过指令中的DI参数指定移动距离，指令发送后等待特定的反馈信息以确保操作成功。
指令替换

    为了实现不同的移动距离，通过正则表达式替换指令中的DI参数。
多轴协调

    通过不同的指令集实现对XYZ三轴的独立控制，并通过调整指令参数实现对不同方向和距离的移动控制。'''
import serial
import time
import re
import copy

X_Electronic_Grearing = 1
Y_Electronic_Grearing = 1
Z_Electronic_Grearing = 1

X_HOM = [
    "1AC100\r",
    "1DE100\r",
    "1VC0.5\r",
    "1VE0.5\r",
    "1FH1\r",
]

Y_HOM = [
    "2AC100\r",
    "2DE100\r",
    "2VC1\r",
    "2VE1\r",
    "2FH2\r",
]


Z_HOM = [
    "3AC100\r",
    "3DE100\r",
    "3VC1\r",
    "3VE1\r",
    "3FH2\r",
]

X_commands = [
    "1AC50\r",
    "1DE50\r",
    "1VE5\r",
    "1VC5\r",
    "1DI2000\r",
    "1FL\r",
]

Y_commands = [
    "2AC50\r",
    "2DE50\r",
    "2VE5\r",
    "2VC5\r",
    "2DI2000\r",
    "2FL\r",
]

Z_commands = [
    "3AC50\r",
    "3DE50\r",
    "3VE5\r",
    "3VC5\r",
    "3DI2000\r",
    "3FL\r",
]

ser = serial.Serial()

def Find_serial():
    ports_list = list(serial.tools.list_ports.comports())
    if len(ports_list) <= 0:
        print("无串口设备。")
    else:
        print("可用的串口设备如下：")
        for comport in ports_list:
            print(list(comport)[0], list(comport)[1])

def serial_init():
    # ser.port='com4'
    ser.port = '/dev/ttyUSB0'
    ser.baudrate=115200
    ser.bytesize=8
    ser.stopbits=1
    ser.timeout=0.1
    ser.parity="N"#奇偶校验位
    ser.open()
    if(ser.isOpen()):
        print("串口打开成功！")
    else:
        print("串口打开失败！")

def serial_close():
    ser.close()
    print("关闭串口")

def X_send_command_with_wait(commands, port='COM3', baudrate=115200, wait_for_1percent=True):
    # try:
        # 打开串口
        # ser = serial.Serial(port, baudrate, timeout=1)
        # print("打开串口")
    for command in commands:
        # 发送指令
        ser.write(command.encode('utf-8'))
        # print("发送指令成功:", command)
        
        # 如果需要等待1%，则等待接收到"1%"的返回值
        if wait_for_1percent and not command.startswith("%"):
            response = ''
            start_time = time.time()  # 记录开始时间
            while '1%' not in response:
                response += ser.read(ser.in_waiting or 1).decode('utf-8')
                if time.time() - start_time > 0.1:  # 超过0.1秒跳出循环
                    break
            # print("接收到回复:", response)

        if command.startswith("%"):
            # 获取 "%" 后面的数
            wait_time = int(command[1:]) / 1000
            # 等待指定秒数
            time.sleep(wait_time)

        # 关闭串口
        # ser.close()
        # print("关闭串口")
    print("X_complete")
    # except Exception as e:
    #     print("发送指令时出错：", e)

def Y_send_command_with_wait(commands, port='COM3', baudrate=115200, wait_for_2percent=True):
    # try:
        # 打开串口
        # ser = serial.Serial(port, baudrate, timeout=1)
        # print("打开串口")
    for command in commands:
        # 发送指令
        ser.write(command.encode('utf-8'))
        # print("发送指令成功:", command)
        
        # 如果需要等待2%，则等待接收到"2%"的返回值
        if wait_for_2percent and not command.startswith("%"):
            response = ''
            start_time = time.time()  # 记录开始时间
            while '2%' not in response:
                response += ser.read(ser.in_waiting or 1).decode('utf-8')
                if time.time() - start_time > 0.1:  # 超过0.1秒跳出循环
                    break
            # print("接收到回复:", response)

        if command.startswith("%"):
            # 获取 "%" 后面的数
            wait_time = int(command[1:]) / 1000
            # 等待指定秒数
            time.sleep(wait_time)

        # 关闭串口
        # ser.close()
        # print("关闭串口")
    print("Y_complete")
    # except Exception as e:
    #     print("发送指令时出错：", e)

def Z_send_command_with_wait(commands, port='COM3', baudrate=115200, wait_for_3percent=True):
    # try:
        # 打开串口
        # ser = serial.Serial(port, baudrate, timeout=1)
        # print("打开串口")
    for command in commands:
        # 发送指令
        ser.write(command.encode('utf-8'))
        # print("发送指令成功:", command)
        
        # 如果需要等待3%，则等待接收到"3%"的返回值
        if wait_for_3percent and not command.startswith("%"):
            response = ''
            start_time = time.time()  # 记录开始时间
            while '3%' not in response:
                response += ser.read(ser.in_waiting or 1).decode('utf-8')
                if time.time() - start_time > 0.1:  # 超过0.1秒跳出循环
                    break
            # print("接收到回复:", response)

            if command.startswith("%"):
                # 获取 "%" 后面的数
                wait_time = int(command[1:]) / 1000
                # 等待指定秒数
                time.sleep(wait_time)

        # 关闭串口
        # ser.close()
        # print("关闭串口")
    print("X_complete")
    # except Exception as e:
    #     print("发送指令时出错：", e)


def replace_DI_data(commands, DI_data):
    tmp = commands.copy()
    for i, command in enumerate(tmp):
        matches = re.findall(r'(\d+)DI(\d*)', command)  # 提取DI前的数字和DI后的数字
        for match in matches:
            di_prefix, di_suffix = match if len(match) == 2 else (match[0], "")
            new_command = f"{di_prefix}DI{DI_data}"
            tmp[i] = tmp[i].replace(f"{di_prefix}DI{di_suffix}", f"{new_command}")
    return tmp

def test_replace_DI_data(distance):
    # distance =-100
    DI_data = int(distance / Z_Electronic_Grearing)
    formatted_DI_data = str(DI_data)
    replace_DI_data(Z_commands, formatted_DI_data)
    print(Z_commands)
# test_replace_DI_data(-1000)
    
def Send_FindHOM():
    Z_send_command_with_wait(Z_HOM)
    time.sleep(1)
    X_send_command_with_wait(X_HOM)
    Y_send_command_with_wait(Y_HOM)
    

# Send_FindHOM()

#Z轴无符号向上，-向下;X轴无符号向左，-向右;Y轴无符号向外，-向里

# X轴运动
def X_Move(distance):
    # X_Electronic_Grearing = 1
    distance = -distance
    DI_data = int(distance / X_Electronic_Grearing)
    formatted_DI_data = str(DI_data)
    tmp = replace_DI_data(X_commands, formatted_DI_data)
    print(tmp)
    X_send_command_with_wait(tmp,wait_for_1percent= True)

# X_Move(70)
# Y轴运动
def Y_Move(distance):
    # X_Electronic_Grearing = 1
    DI_data = int(distance / Y_Electronic_Grearing)
    formatted_DI_data = str(DI_data)
    tmp = replace_DI_data(Y_commands, formatted_DI_data)
    print(tmp)
    Y_send_command_with_wait(tmp,wait_for_2percent= True)

# Y_Move(400)
# Z轴运动
def Z_Move(distance):
    # X_Electronic_Grearing = 1
    DI_data = int(distance / Z_Electronic_Grearing)
    formatted_DI_data = str(DI_data)
    tmp =  replace_DI_data(Z_commands, formatted_DI_data)
    print(tmp)
    # print(Z_commands)
    Z_send_command_with_wait(tmp,wait_for_3percent = True)

# serial_init()
# X_Move(50)
# Z_Move(-100)
# Z_send_command_with_wait(Z_HOM,wait_for_3percent=False)
# # time.sleep(2)
# Z_Move(2000)
# X_send_command_with_wait(X_HOM)
# X_Move(60)
# # time.sleep(2)
# Z_Move(1000)
# serial_close()