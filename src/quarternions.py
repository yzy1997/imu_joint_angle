import numpy as np
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2euler

# 假设IMU9的采集频率为100Hz
sample_rate = 52.0  # 100Hz
madgwick = Madgwick(frequency=sample_rate)

# 示例加速度、角速度和磁力计数据 (单位: 加速度 m/s², 角速度 rad/s, 磁力计 µT)
accel_data = np.array([0.0, 9.81, 0.0])  # 3D 加速度数据
gyro_data = np.array([0.01, 0.02, 0.015])  # 3D 角速度数据
mag_data = np.array([30.0, 50.0, -40.0])  # 3D 磁力计数据

# 初始化四元数（单位四元数）
q = np.array([1.0, 0.0, 0.0, 0.0])

# 使用Madgwick滤波器进行更新，生成新的四元数
q = madgwick.updateMARG(q, gyr=gyro_data, acc=accel_data, mag=mag_data)

# 输出生成的四元数
print("Estimated Quaternion:", q)

# 可选：将四元数转换为欧拉角（弧度）
euler_angles = q2euler(q)
print("Euler Angles (radians):", euler_angles)

# 如果需要转换为度：
euler_angles_degrees = np.degrees(euler_angles)
print("Euler Angles (degrees):", euler_angles_degrees)
