'''### 电机控制模块技术文档 ####

---

#### 概述 ####

本模块通过集成`m_ModBusTcp`和`m_RS485`两个子模块
提供了一套完整的电机初始化 定位 移动和检测操作的API接口
实现对电机的多轴控制XYZ以及对设备的归零和半圆旋转操作

---

#### API接口 ####

1. **motor_init()**
   - **功能**: 初始化电机 包括连接设备和归零操作
   - **输入**: 无
   - **输出**: 无

2. **motor_hom()**
   - **功能**: 对所有电机进行归零操作
   - **输入**: 无
   - **输出**: 无

3. **motor_locate()**
   - **功能**: 将电机移动到初始位置 X轴左移55 Y轴外移2100 待定
   - **输入**: 无
   - **输出**: 无

4. **motor_close()**
   - **功能**: 关闭与电机的连接
   - **输入**: 无
   - **输出**: 无

5. **G_inward_4mm()**
   - **功能**: 控制G设备向内移动4mm
   - **输入**: 无
   - **输出**: 无

6. **G_out_4mm()**
   - **功能**: 控制G设备向外移动4mm
   - **输入**: 无
   - **输出**: 无

7. **R_rotate_Counterclockwise_semi_circle()**
   - **功能**: 控制R设备逆时针旋转半圈
   - **输入**: 无
   - **输出**: 无

8. **R_rotate_clockwise_semi_circle()**
   - **功能**: 控制R设备顺时针旋转半圈
   - **输入**: 无
   - **输出**: 无

9. **R_rotate_Counterclockwise_quarter_circle()**
   - **功能**: 控制R设备逆时针旋转1/4圈
   - **输入**: 无
   - **输出**: 无

10. **R_rotate_clockwise_quarter_circle()**
    - **功能**: 控制R设备顺时针旋转1/4圈
    - **输入**: 无
    - **输出**: 无

11. **X_move_right_one_space()**
    - **功能**: 控制X轴右移一格73单位
    - **输入**: 无
    - **输出**: 无

12. **X_move_left_one_space()**
    - **功能**: 控制X轴左移一格73单位
    - **输入**: 无
    - **输出**: 无

13. **X_Find_Camera(X_Number)**
    - **功能**: 根据X轴编号移动到相机位置。
    - **输入**: `X_Number`(int): X轴编号
    - **输出**: 无

14. **X_Find_Tube(X_Number)**
    - **功能**: 根据X轴编号移动到管道位置。
    - **输入**: `X_Number`(int): X轴编号
    - **输出**: 无

15. **Y_move_forward_one_space()**
    - **功能**: 控制Y轴前移一格420单位
    - **输入**: 无
    - **输出**: 无

16. **Y_move_back_one_space()**
    - **功能**: 控制Y轴后移一格420单位
    - **输入**: 无
    - **输出**: 无

17. **Y_move_forward_shelf()**
    - **功能**: 控制Y轴前移一个架子位置1680单位
    - **输入**: 无
    - **输出**: 无

18. **Z_move_up_one_space()**
    - **功能**: 控制Z轴上移一格2230单位
    - **输入**: 无
    - **输出**: 无

19. **Z_move_down_one_space()**
    - **功能**: 控制Z轴下移一格2230单位
    - **输入**: 无
    - **输出**: 无

20. **Z_Find_Camera()**
    - **功能**: 将Z轴移动到相机位置
    - **输入**: 无
    - **输出**: 无

21. **Z_Find_Tube()**
    - **功能**: 将Z轴移动到管道位置
    - **输入**: 无
    - **输出**: 无

22. **detect_once(X_Number)**
    - **功能**: 执行一次完整的检测流程 包括Z轴 X轴
                G设备和R设备的多次移动和旋转操作
    - **输入**: `X_Number`(int): X轴编号
    - **输出**: 无

23. **detect_all()**
    - **功能**: 执行一次完整的批量检测流程
                包括多次调用`detect_once`函数 并控制电机的定位和归零操作
    - **输入**: 无
    - **输出**: 无

---

#### 功能原理 ####

1. **电机初始化**
   - 通过调用`m_ModBusTcp`模块和`m_RS485`模块的初始化函数 连接到设备并执行归零操作

2. **电机定位**
   - 通过控制电机的多轴移动将电机移动到指定位置
   - 初始定位包括X轴左移55单位和Y轴前移2100单位

3. **多轴移动**
   - 通过调用`m_RS485`模块的移动函数控制XYZ轴的移动
   - 每个移动操作通过指定移动距离，实现电机的精确定位

4. **设备旋转**
   - 通过调用`m_ModBusTcp`模块的旋转函数控制R设备的半圆和四分之一圆的旋转操作

5. **检测流程**
   - 通过调用`detect_once`函数执行一次完整的检测流程
   - 包括Z轴下降 X轴移动 G设备夹持和R设备旋转操作
   - 通过线程事件机制`stop_event`实现流程的同步和等待操作

6. **批量检测**
   - 通过调用`detect_all`函数执行一次完整的批量检测流程循环调用`detect_once`函数
   - 并控制X轴和Y轴的移动 实现批量检测操作

---

通过上述API接口和功能原理本模块实现了对电机的多轴控制和检测操作 确保设备在自动化流程中的精确性和稳定性。'''

import motor.m_ModBusTcp as m_ModBusTcp
import motor.m_RS485 as m_RS485
import time
import threading

stop_event = threading.Event()
batch = 1

def motor_init():
    m_ModBusTcp.connect_init()
    m_ModBusTcp.G_Send_HOM()
    m_ModBusTcp.R_Send_HOM()
    m_RS485.serial_init()
    m_RS485.Send_FindHOM()

def motor_hom():
    m_ModBusTcp.G_Send_HOM()
    m_ModBusTcp.R_Send_HOM()
    m_RS485.Send_FindHOM()


#归零后x轴左移60，y轴外移2100,Z轴下移1000
def motor_locate():
    m_RS485.X_Move(-55)
    m_RS485.Y_Move(2100)
    # m_RS485.Z_Move(-200)


def motor_close():
    m_ModBusTcp.connect_close()
    m_RS485.serial_close()

def G_inward_4mm():
    m_ModBusTcp.G_Send_RC1()

def G_out_4mm():
    m_ModBusTcp.G_Send_RC2()

def R_rotate_Counterclockwise_semi_circle():
    m_ModBusTcp.R_Send_RC1()

def R_rotate_clockwise_semi_circle():
    m_ModBusTcp.R_Send_RC2()

#x轴每格间距70
def X_move_right_one_space():
    m_RS485.X_Move(143)

def X_move_left_one_space():
    m_RS485.X_Move(-143)

def X_Find_Camera(X_Number):
    tmp = 1130-143*X_Number
    m_RS485.X_Move(tmp)

def X_Find_Tube(X_Number):
    tmp = 143*X_Number-1130
    m_RS485.X_Move(tmp)

#Y轴每格间距400
def Y_move_forward_one_space():
    m_RS485.Y_Move(420)

def Y_move_back_one_space():
    m_RS485.Y_Move(-420)

def Y_move_forward_shelf():
    m_RS485.Y_Move(1680)

#Z轴上下移动2000
def Z_move_up_one_space():
    m_RS485.Z_Move(2230)

def Z_move_down_one_space():
    m_RS485.Z_Move(-2230)

def Z_Find_Camera():
    m_RS485.Z_Move(-2600)

def Z_Find_Tube():
    m_RS485.Z_Move(2600)


def detect_once(X_Number):
    Z_move_down_one_space()
    time.sleep(3)
    stop_event.wait()
    G_inward_4mm()
    stop_event.wait()
    Z_move_up_one_space()
    time.sleep(3)
    stop_event.wait()
    X_Find_Camera(X_Number)
    time.sleep(2)
    stop_event.wait()
    Z_Find_Camera()
    time.sleep(2.5)
    stop_event.wait()
    R_rotate_Counterclockwise_semi_circle()
    stop_event.wait()
    R_rotate_Counterclockwise_semi_circle()
    stop_event.wait()
    # rotate_event.wait()
    Z_Find_Tube()
    time.sleep(3.5)
    stop_event.wait()
    X_Find_Tube(X_Number)
    time.sleep(2)
    stop_event.wait()
    Z_move_down_one_space()
    time.sleep(2)
    stop_event.wait()
    G_out_4mm()
    stop_event.wait()
    Z_move_up_one_space()
    time.sleep(2)
    stop_event.wait()


def detect_all():
    global batch
    while True:
        stop_event.wait()
        # set right after self.motor_control_thread.daemon = True
        motor_locate()
        time.sleep(2)
        Y_Number = 0
        X_Number = 0

        while X_Number < 2:
            if Y_Number == 4:
                detect_once(X_Number)
                X_move_right_one_space()
                stop_event.wait()
                Y_move_forward_shelf()
                stop_event.wait()
                Y_Number = 0
                X_Number += 1
            else :
                detect_once(X_Number)
                Y_move_back_one_space()
                stop_event.wait()
                Y_Number += 1

        motor_hom()
        batch += 1

        stop_event.clear()
    

# motor_init()
# time.sleep(15)
# detect_all()
# motor_close()