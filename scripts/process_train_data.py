import torch
import torchvision
import numpy as np
import os
import json
from PIL import Image
from resnet_utils import myResnet

operation_log='../训练数据样本'
if not os.path.exists(operation_log):
   os.makedirs(operation_log)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
resnet101=torchvision.models.resnet101(pretrained=True).eval()
resnet101=myResnet(resnet101).cuda(device).requires_grad_(False)
词数词典路径="./json/词_数表.json"

with open(词数词典路径, encoding='utf8') as f:
    词数词典=json.load(f)

for root, dirs, files in os.walk(operation_log):
    if len(dirs)>0:
        break
for 号 in dirs:
    路径json = operation_log+'/' + 号 + '/_操作数据.json'
    numpy数组路径= operation_log+'/' + 号 + '/图片_操作预处理数据2.npz'
    if os.path.isfile(numpy数组路径):
        continue

    img_tensor = torch.Tensor(0)

    # print(img_tensor.shape[0])
    操作张量 = torch.Tensor(0)

    伪词序列 = torch.from_numpy(np.ones((1, 60)).astype(np.int64)).cuda(device).unsqueeze(0)

    操作序列 = np.ones((1, 1))
    计数 = 0
    print('正在处理{}'.format(号))
    数据列=[]
    with open(路径json, encoding='ansi') as f:
        移动操作='无移动'
        while True:
            df = f.readline()


            if df == "":
                break
            df = json.loads(df)
            数据列.append(df)
    # for i in range(len(数据列)):
    #     if i>0 and 数据列[i]['动作操作']!='无动作' and 数据列[i-1]['动作操作']=='无动作' :
    #         数据列[i-1]['动作操作']=数据列[i]['动作操作']
    #         if i>1 and  数据列[i-2]['动作操作']=='无动作' :
    #             数据列[i - 2]['动作操作'] = 数据列[i]['动作操作']



    with open(路径json, encoding='ansi') as f:
        移动操作='无移动'
        for i in range(len(数据列)):
            df = 数据列[i]

            if img_tensor.shape[0] == 0:
                img = Image.open(operation_log+'/' + 号 + '/{}.jpg'.format(df["图片号"]))
                img2 = np.array(img)

                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _,out = resnet101(img2)
                img_tensor = out.reshape(1,6*6*2048)
                移动操作a=df["移动操作"]
                if 移动操作a!='无移动':
                    移动操作=移动操作a

                操作序列[0, 0] = 词数词典[移动操作 + "_" + df["动作操作"]]
            else:
                img = Image.open(operation_log+'/' + 号 + '/{}.jpg'.format(df["图片号"]))
                img2 = np.array(img)

                img2 = torch.from_numpy(img2).cuda(device).unsqueeze(0).permute(0, 3, 2, 1) / 255
                _,out= resnet101(img2)

                img_tensor = torch.cat((img_tensor, out.reshape(1,6*6*2048)), 0)
                移动操作a=df["移动操作"]
                if 移动操作a!='无移动':
                    移动操作=移动操作a
                操作序列=np.append(操作序列, 词数词典[移动操作 + "_" + df["动作操作"]])
                #操作序列[0, 0] = 词数词典[df["移动操作"] + "_" + df["动作操作"]]

        img_tensor_np=img_tensor.cpu().numpy()
        操作序列=操作序列.astype(np.int64)
        np.savez(numpy数组路径, img_tensornp=img_tensornp, 操作序列=操作序列)

