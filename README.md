# honor ai

---

# 1.功能描述
使用ai 玩王者荣耀，整合了深度学习玩王者和强化学习玩王者，优化了代码中的中文，适配的ubuntu20.04环境，后续适配到docker中

---

# 2.依赖

## 2.1 scripy

说明： 将手机镜像到电脑中，方便操作
```
sudo apt install scripy
```

## 2.2 其他见requiremets

```bash
torch==1.10.2+cu111
torchvision==0.11.3+cu111
torchaudio==0.10.2+cu111
-f https://download.pytorch.org/whl/cu111/torch_stable.html
pyminitouch
pynput
```

## 2.3 模型文件

百度网盘 链接：https://pan.baidu.com/s/1Ak1sLcSRimMWRgagXGahTg 提取码：t4k3

## 2.4 训练数据

百度网盘 链接：todo 提取码：todo

# 3.运行

## 3.1 train

将图片用resnet101预处理后再和对应操作数据一起处理后用numpy数组储存备用

```bash
python process_train_data.py
```

训练
```bash
训练
python train.py
```

## 3.2 test

# 其他

参考： 
https://github.com/FengQuanLi/ResnetGPT

https://github.com/FengQuanLi/WZCQ


