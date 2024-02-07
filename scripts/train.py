import torch
import torchvision
from PIL import Image
import numpy as np
import time
import json
from model.config import GPT2Config, TransformerConfig
from model.batch import create_masks
from model.model import get_model
import torch.nn.functional as F
from scripts.get_train_data import *
from scripts.utils import *
import os
import random

# Training data saving directory
train_data_save_directory = "../train_data_samples"

# If the directory does not exist, create it
if not os.path.exists(train_data_save_directory):
    os.makedirs(train_data_save_directory)

# For loop for root, directories, and files
for root, dirs, files in os.walk("../train_data_samples"):
    if len(dirs) > 0:
        break

# Word count dictionary path
word_count_dict_path = "./json/词_数表.json"

# Number-word table path
num_word_table_path = "./json/数_词表.json"

# If the word_count_dict_path and num_word_table_path exist
if os.path.isfile(word_count_dict_path) and os.path.isfile(num_word_table_path):
    with open(word_count_dict_path, "r") as f:
        word_count_dict = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the TransformerConfig
    config = TransformerConfig()

    # Get the model
    model = get_model(config, 130)

    # Load the model weights
    model = model.cuda(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=6.25e-5, betas=(0.9, 0.98), eps=1e-9)

    # Model path
    model_path = "weights/model_weights"

    # Model weights saving directory
    model_weights_save_directory = "weights/model_weights"

    # Count
    count = 0

    # Time start
    time_start = time.time()

    # For loop for j
    for j in range(100):
        # Randomly shuffle directories
        random.shuffle(dirs)

        # For loop for each directory
        for _ in dirs:
            #预处理数据
            preprocessing_data = "../train_data_samples/" + _ + "/图片_操作预处理数据2.npz"
            if os.path.isfile(preprocessing_data):
                npz_file = np.load(preprocessing_data, allow_pickle=True)
                npz_data = npz_file["图片张量np"]
                npz_data = np.insert(npz_data, 0, 128)

                # Loop for loop
                loop = True
                cursor = 0
                # Cursor size
                cursor_size = 23
                # Branch size
                branch = 10

                # Table for saving the divided data
                operation_division_table = []
                # Table for saving the target output
                target_output_table = []
                # Table for saving the image
                image_table = []

                # While loop
                while loop:
                    # If cursor + cursor_size < npz_data.shape[0]:
                    if cursor + cursor_size < len(npz_data):
                        # Division operation
                        operation_division = npz_data[cursor : cursor + cursor_size]
                        # Target output
                        target_output = npz_data[cursor + 1 : cursor + 1 + cursor_size]
                        # Image
                        image = npz_data[cursor + cursor_size + 1 : cursor + cursor_size + 1 + cursor_size, :]
                        # Append to the table
                        operation_division_table.append(operation_division)
                        target_output_table.append(target_output)
                        image_table.append(image)
                        # Cursor increase
                        cursor = cursor + cursor_size
                    else:
                        # Division operation
                        operation_division = npz_data[-cursor_size - 1 : -1]
                        # Target output
                        target_output = npz_data[-cursor_size:]
                        # Image
                        image = npz_data[-cursor_size - 1 - cursor_size : -cursor_size - 1, :]
                        # Append to the table
                        operation_division_table.append(operation_division)
                        target_output_table.append(target_output)
                        image_table.append(image)
                        # Loop end
                        loop = False

                # Loop for loop
                i = 0
                # Loop continue
                loop = True
                # While loop
                while loop:
                    # If i + 1 * branch < len(operation_division_table):
                    if i + 1 * branch < len(operation_division_table):
                        # Division operation
                        operation_division_division = np.array(operation_division_table[i * branch : (i + 1) * branch])
                        # Image
                        image_division = np.array(image_table[i * branch : (i + 1) * branch])
                        # Target output
                        target_output_division = np.array(target_output_table[i * branch : (i + 1) * branch])

                    else:
                        # Division operation
                        operation_division_division = np.array(operation_division_table[i * branch : len(operation_division_table)])
                        # Image
                        image_division = np.array(
                            image_table[i * branch : len(image_table)], dtype=np.float32
                        )
                        # Target output
                        target_output_division = np.array(
                            target_output_table[i * branch : len(target_output_table)]
                        )
                        # Loop end
                        loop = False

                    # Division operation table torch
                    operation_division_torch = torch.from_numpy(operation_division_division).cuda(device)
                    # Image table torch
                    image_division_torch = torch.from_numpy(image_division).cuda(device)
                    # Target output table torch
                    target_output_division_torch = torch.from_numpy(target_output_division).cuda(device)

                    # Create masks for the division operation, image, and target output
                    src_mask, trg_mask = create_masks(operation_division_torch, operation_division_torch, device)
                    if image_division_torch.shape[0]!= operation_division_torch.shape[0]:
                        continue
                    # Output_实际_A
                    output_实际_A = model(image_division_torch, operation_division_torch, trg_mask)
                    # Output_实际_A torch
                    output_实际_A_torch = output_实际_A.view(-1, output_实际_A.size(-1))
                    # Zero grad
                    optimizer.zero_grad()
                    # Cross entropy loss
                    loss = F.cross_entropy(output_实际_A_torch, target_output_division_torch.contiguous().view(-1), ignore_index=-1)
                    # If count % 1 == 0:
                    if count % 1 == 0:
                        print(loss)

                    # Calculate the time difference
                    time_end = time.time()
                    # Calculate the time used
                    time_cost = time_end - time_start

                    # If count % 45060 == 0:
                    if count % 45060 == 0:
                        print("888")

                    # Loss backward
                    loss.backward()

                    # Step optimizer
                    optimizer.step()

                    # Count increase
                    count = count + 1
                    # i increase
                    i = i + 1
    torch.save(model.state_dict(), "weights/model_weights")
    torch.save(model.state_dict(), "weights/model_weights_P{}".format(str(j)))