import logging
import cv2
import numpy as np
import datetime

# 假设这是客户端中的环境接口，用来获取图像帧
def get_frames():
    # 这里是模拟从环境中获取图像的过程
    # 实际情况你可以从 robot_env 或其他方式获取图像帧
    return {
        "image_right": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),  # 模拟右侧相机图像
        "image_left": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),   # 模拟左侧相机图像
        "image_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8)   # 模拟腕部相机图像
    }

# 假设这是客户端发送数据的地方
def main():
    frames = get_frames()  # 获取图像帧

    # 打印每个图像的形状
    for server_name, img in frames.items():
        print(f"Received image '{server_name}' with shape: {img.shape}")

        # 保存图像用于调试
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = f"./log_images/{server_name}_{timestamp}.png"
        cv2.imwrite(log_path, img)
        print(f"Image saved at {log_path}")

if __name__ == "__main__":
    main()


# 5.13 号测试的时候的状态 [-0.0568, 1.5278, -0.925, 0.1212, -0.0506, -0.0874]