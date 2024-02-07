from torch.autograd import Variable
import torch
import numpy as np

def print_sampled_data(word_count_table, data, output_scores):
    instance = data[0]
    printable = [word_count_table[str(instance[i, 0])] for i in range(0, instance.shape[0])]
    instance = output_scores.cpu().numpy()
    printable2 = [word_count_table[str(instance[i])] for i in range(0, instance.shape[0])]
    print("Sampled Output", printable)
    print("Target Output", printable2)
    # for i in range(16):# print(word_count_table[str(instance[i, 0])])

def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)),
    k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)

    np_mask = np_mask.cuda(device)
    return np_mask


def print_test_data(word_count_table, data, input_scores, labels):
    instance = data[0]
    printable = [word_count_table[str(instance[i])] for i in range(instance.size)]
    print_data = ""
    for i in range(len(printable)):
        print_data += printable[i]
    instance = input_scores.cpu().numpy()[0]
    printable2 = [word_count_table[str(instance[i])] for i in range(input_scores.size(1))]
    # printable2 = str(printable2)
    # print("Input:", printable2)
    if labels == print_data:
        return True
    else:
        return False


def print_test_data_a(word_count_table, data, input_scores):
    if data.shape[0]!= 0:
        instance = data[0]
        printable = [word_count_table[str(instance[i])] for i in range(instance.size)]
        print_data = ""
        for i in range(len(printable)):
            print_data = print_data + printable[i]
        instance = input_scores.cpu().numpy()[0]
        printable2 = [word_count_table[str(instance[i])] for i in range(input_scores.size(1))]
        printable2 = str(printable2)
        # print("Input:", printable2)
        print("Output:", print)
