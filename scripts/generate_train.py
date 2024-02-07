import os
import time
import torchvision
from model.config import GPT2Config, TransformerConfig
from model.batch import create_masks
from model.model import get_model
import torch.nn.functional as F
from get_train_data import *
from scripts.utils import *
import random
from resnet_utils import myResnet
from run_assist import *
from pynput.keyboard import Controller, Key, Listener
from pynput import keyboard
import time, threading

_DEVICE_ID = "c50f679f"
window_name = "RNE-AL00"
model = "model_weights_O35"
train_dir = "/media/hh/Windows/honer_of_kings"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
lock = threading.Lock()
start = time.time()
end = time.time()
fun_start = 0
time_interval = 0
index = 0
dict = {"interval_times": 0, "max_interval": 0.0, "interval_location": []}
count = 0
count_dict = {"first_time": 0.0, "first_p_to_second_r": 0.0}
keyBoard_dict = {"Key.enter": "\n", "Key.space": " ", "Key.tab": "\t"}

press_w = False
press_s = False
press_a = False
press_d = False
press_q = False
attack_status = False
manual_mode = False
attack_release = True
ai_on = True
operation_col = []


def get_key_name(key):
    if isinstance(key, keyboard.KeyCode):
        return key.char
    else:
        return str(key)


# 监听按压
def on_press(key):
    global fun_start, time_interval, index, dict, count, count_dict, press_w, press_s, press_a, press_d, manual_mode, operation_col, ai_on, attack_release, press_q, attack_status

    key_name = get_key_name(key)
    operation = ""
    if key_name == "w":
        press_w = True
    elif key_name == "a":
        press_a = True
    elif key_name == "s":
        press_s = True
    elif key_name == "d":
        press_d = True
    elif key_name == "q":
        press_q = True
    elif key_name == "i":
        ai_on = not ai_on

    elif key_name == "Key.space":
        operation = "召唤师技能"
    elif key_name == "Key.end":
        operation = "补刀"
    elif key_name == "Key.page_down":
        operation = "推塔"
    elif key_name == "j":
        operation = "一技能"
    elif key_name == "k":
        operation = "二技能"
    elif key_name == "l":
        operation = "三技能"
    elif key_name == "f":
        operation = "回城"
    elif key_name == "g":
        operation = "恢复"
    elif key_name == "h":
        operation = "召唤师技能"
    elif key_name == "Key.left":
        operation = "一技能"
    elif key_name == "Key.down":
        operation = "二技能"
    elif key_name == "Key.right":
        operation = "三技能"
    elif key_name == "Key.up":
        attack_status = True

    lock.acquire()
    if operation != "":
        operation_col.append(operation)
    lock.release()
    # print("正在按压:", key_name)


# 监听释放
def on_release(key):
    global start, fun_start, time_interval, index, count, count_dict, press_w, press_s, press_a, press_d, attack_release, press_q, attack_status

    key_name = get_key_name(key)
    if key_name == "w":
        press_w = False
    elif key_name == "a":
        press_a = False
    elif key_name == "s":
        press_s = False
    elif key_name == "d":
        press_d = False
    elif key_name == "q":
        press_q = False

    elif key_name == "Key.up":
        attack_status = False
    print("已经释放:", key_name)
    if key == Key.esc:
        # 停止监听
        return False


# 开始监听
def start_listen():
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()


def process_direction():
    # press_w = False
    # press_s = False
    # press_a = False
    # press_d = False
    if press_q == True:
        return "移动停"
    elif press_w == True and press_s == False and press_a == False and press_d == False:
        return "上移"
    elif press_w == False and press_s == True and press_a == False and press_d == False:
        return "下移"
    elif press_w == False and press_s == False and press_a == True and press_d == False:
        return "左移"
    elif press_w == False and press_s == False and press_a == False and press_d == True:
        return "右移"
    elif press_w == True and press_s == False and press_a == True and press_d == False:
        return "左上移"
    elif press_w == True and press_s == False and press_a == False and press_d == True:
        return "右上移"
    elif press_w == False and press_s == True and press_a == True and press_d == False:
        return "左下移"
    elif press_w == False and press_s == True and press_a == False and press_d == True:
        return "右下移"
    else:
        return ""


upgrade_3rd_skill = "d 0 552 1878 100\nc\nu 0\nc\n"
upgrade_2nd_skill = "d 0 446 1687 100\nc\nu 0\nc\n"
upgrade_1st_skill = "d 0 241 1559 100\nc\nu 0\nc\n"
shopping = "d 0 651 207 100\nc\nu 0\nc\n"
cmd_id_path = "./config/cmd_id.json"
id_cmd_path = "./config/id_cmd.json"
operation查询路径 = "./json/名称_operation.json"
operation词典 = {"图片号": "0", "移动operation": "无移动", "动作operation": "无动作"}
th = threading.Thread(
    target=start_listen,
)
th.start()  # 启动线程

if os.path.isfile(id_cmd_path) and os.path.isfile(cmd_id_path):
    cmd_id, id_cmd = read_index(cmd_id_path, id_cmd_path)
with open(id_cmd_path, encoding="utf8") as f:
    id_cmd_dict = json.load(f)
with open(operation查询路径, encoding="utf8") as f:
    operation查询词典 = json.load(f)

方向表 = ["上移", "下移", "左移", "右移", "左上移", "左下移", "右上移", "右下移"]


device = MyMNTDevice(_DEVICE_ID)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
mod = (
    torchvision.models.resnet101(pretrained=True)
    .eval()
    .cuda(device)
    .requires_grad_(False)
)
resnet101 = myResnet(mod)
config = TransformerConfig()

model = get_model(config, 130, model)

model = model.cuda(device).requires_grad_(False)

while True:
    if ai_on:

        img_path = train_dir + "/{}/".format(str(int(time.time())))
        os.mkdir(img_path)

        record_file = open(img_path + "_operation_data.json", "w+")

        img_tensor = torch.Tensor(0)
        operation_tensor = torch.Tensor(0)

        pseudo_word_seq = (
            torch.from_numpy(np.ones((1, 60)).astype(np.int64))
            .cuda(device)
            .unsqueeze(0)
        )

        cmd_delay = 0

        operation_seq = np.ones((1,))
        operation_seq[0] = 128
        counter = 0
        time_start = time.time()
        pre_cmd = "移动停"
        for i in range(1000000):
            if ai_on == False:
                break
            try:
                imgA = get_img(window_name)
            except:
                ai_on = False
                print("取图失败")
                break

            start_timing = time.time()

            if img_tensor.shape[0] == 0:

                img = np.array(imgA)

                img = (
                    torch.from_numpy(img).cuda(device).unsqueeze(0).permute(0, 3, 2, 1)
                    / 255
                )
                _, out = resnet101(img)
                img_tensor = out.reshape(1, 6 * 6 * 2048)

            elif img_tensor.shape[0] < 19:

                img = np.array(imgA)

                img = (
                    torch.from_numpy(img).cuda(device).unsqueeze(0).permute(0, 3, 2, 1)
                    / 255
                )
                _, out = resnet101(img)
                img_tensor = torch.cat((img_tensor, out.reshape(1, 6 * 6 * 2048)), 0)
                operation_seq = np.append(operation_seq, 抽样np[0, 0])

            else:

                img = np.array(imgA)

                img = (
                    torch.from_numpy(img).cuda(device).unsqueeze(0).permute(0, 3, 2, 1)
                    / 255
                )
                _, out = resnet101(img)
                img_tensor = img_tensor[0:18, :]
                operation_seq = operation_seq[0:18]
                operation_seq = np.append(operation_seq, 抽样np[0, 0])
                img_tensor = torch.cat((img_tensor, out.reshape(1, 6 * 6 * 2048)), 0)

            operation_tensor = torch.from_numpy(operation_seq.astype(np.int64)).cuda(device)
            src_mask, trg_mask = create_masks(
                operation_tensor.unsqueeze(0), operation_tensor.unsqueeze(0), device
            )
            输出_实际_A = model(img_tensor.unsqueeze(0), operation_tensor.unsqueeze(0), trg_mask)

            LI = operation_tensor.contiguous().view(-1)
            # LA=输出_实际_A.view(-1, 输出_实际_A.size(-1))
            if counter % 50 == 0 and counter != 0:

                device.send(shopping)
                device.send(upgrade_3rd_skill)
                device.send(upgrade_2nd_skill)
                device.send(upgrade_1st_skill)
                device.send(operation查询词典["移动停"])
                print(pre_cmd, "周期")
                time.sleep(0.02)
                device.send(operation查询词典[pre_cmd])

            if counter % 1 == 0:
                time_end = time.time()

                输出_实际_A = F.softmax(输出_实际_A, dim=-1)
                输出_实际_A = 输出_实际_A[:, -1, :]
                抽样 = torch.multinomial(输出_实际_A, num_samples=1)
                抽样np = 抽样.cpu().numpy()

                指令 = 数_词表[str(抽样np[0, -1])]
                指令集 = 指令.split("_")

                # operation词典 = {"图片号": "0", "移动operation": "无移动", "动作operation": "无动作"}
                operation词典["图片号"] = str(i)
                方向结果 = process_direction()
                if 方向结果 != "" or len(operation_col) != 0 or attack_status == True:
                    if 方向结果 == "":
                        operation词典["移动operation"] = 指令集[0]
                    else:
                        operation词典["移动operation"] = 方向结果

                    if len(operation_col) != 0:
                        operation词典["动作operation"] = operation_col[0]
                        lock.acquire()
                        del operation_col[0]
                        lock.release()
                    elif attack_status == True:
                        operation词典["动作operation"] = "攻击"

                    else:
                        operation词典["动作operation"] = "无动作"

                    路径_a = img_path + "{}.jpg".format(str(i))
                    imgA.save(路径_a)
                    json.dump(operation词典, record_file, ensure_ascii=False)
                    record_file.write("\n")

                    新指令 = operation词典["移动operation"]
                    if 新指令 != pre_cmd and 新指令 != "无移动":
                        pre_cmd = 新指令
                        # print(pre_cmd,operation查询词典[pre_cmd])
                        try:
                            print("manual_mode", pre_cmd)

                            device.send(operation查询词典[pre_cmd])

                        except:
                            ai_on = False
                            print("send失败")
                            break

                        time.sleep(0.01)

                    if (
                        operation词典["动作operation"] != "无动作"
                        and operation词典["动作operation"] != "发起集合"
                        and operation词典["动作operation"] != "发起进攻"
                        and operation词典["动作operation"] != "发起撤退"
                    ):
                        print("手动", 指令集[1])
                        try:
                            device.send(operation查询词典[operation词典["动作operation"]])
                        except:
                            ai_on = False
                            print("send失败")
                            break
                else:
                    operation_col = []
                    operation词典["移动operation"] = 指令集[0]
                    operation词典["动作operation"] = 指令集[1]

                    新指令 = 指令集[0]
                    if 新指令 != pre_cmd and 新指令 != "无移动":
                        pre_cmd = 新指令
                        # print(pre_cmd,operation查询词典[pre_cmd])
                        try:
                            print(pre_cmd)

                            device.send(operation查询词典[pre_cmd])

                        except:
                            ai_on = False
                            print("send失败")
                            break

                        time.sleep(0.01)

                    if (
                        指令集[1] != "无动作"
                        and 指令集[1] != "发起集合"
                        and 指令集[1] != "发起进攻"
                        and 指令集[1] != "发起撤退"
                    ):
                        print(指令集[1])
                        try:
                            device.send(operation查询词典[指令集[1]])
                        except:
                            ai_on = False
                            print("send失败")
                            break
                用时1 = 0.22 - (time.time() - start_timing)
                if 用时1 > 0:
                    time.sleep(用时1)

                用时 = time_end - time_start
                # print("用时{} 第{}张 延时{}".format(用时, i,用时1),'press_a', press_a, 'press_w', press_w, 'press_s', press_s, 'press_d', press_d, 'pre_cmd', pre_cmd, 'ai_on', ai_on, 'operation_col', operation_col)

                counter = counter + 1

    record_file.close()
    time.sleep(1)
    print("ai_on", ai_on)
