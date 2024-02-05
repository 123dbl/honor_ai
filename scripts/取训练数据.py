import json
import numpy as np


def read_train_data(path):
    inputs = []
    ouputs = []
    with open(path, encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            json_line = json.loads(line)

            content = json_line["content______"]
            content_input = content["inputs______"]
            content_output = content["outputs______"]
            # 这里的数据还得进行分割先暂时分割成16份吧
            unit_length = len(content_input) // 16
            for i in range(16):
                # print(内容_输入[i*单元长度:(i+1)*单元长度])
                inputs.append(content_input[i * unit_length : (i + 1) * unit_length])
                ouputs.append(content_output[i * unit_length : (i + 1) * unit_length])
    return inputs, ouputs


def write_word_index(total_word_table, word_count_table_path, char_count_table_path):
    print("Writing word index data, which may take a long time.")
    index_to_char = {}
    char_to_index = {}
    index_chars = []

    # index_to_char = list(set(total_word_table))
    i = 0
    j = 0
    for word_table in total_word_table:
        j = j + 1
        for char in word_table:

            if char not in index_chars:
                index_chars.append(char)
                char_to_index[char] = i
                index_to_char[i] = char
                i = i + 1
        if j % 10000 == 0:
            print(i, index_to_char[i - 1], j / len(total_word_table))

    # print(index_to_char[1], index_to_char[111], len(index_to_char))
    with open(word_count_table_path, "w", encoding="utf-8") as f:
        json.dump(char_to_index, f, ensure_ascii=False)
    with open(char_count_table_path, "w", encoding="utf-8") as f:
        json.dump(index_to_char, f, ensure_ascii=False)


def read_index(word_count_table_path, char_count_table_path):
    with open(word_count_table_path, encoding="utf-8") as f:
        word_count_table = json.load(f)

    with open(char_count_table_path, encoding="utf-8") as f:
        char_count_table = json.load(f)
    return word_count_table, char_count_table


def generate_training_arrays(input_data, word_count_table, numpy_array_path):
    table_1 = []
    table_2 = []
    i = 0
    context = ""
    for form in input_data:
        table_3 = []
        for char in form:
            if (ord('a') <= ord(char) <= ord('z')) or (ord('A') <= ord(char) <= ord('Z')):
                if context == "":
                    context = char
                else:
                    context += char
            else:
                if context == "":
                    if char.lower() in word_count_table:
                        table_3.append(word_count_table[char.lower()])
                    else:
                        table_3.append(14999)
                else:
                    if context.lower() in word_count_table:
                        table_3.append(word_count_table[context.lower()])
                    else:
                        table_3.append(14999)
                    context = ""
                    if char.lower() in word_count_table:
                        table_3.append(word_count_table[char.lower()])
                    else:
                        table_3.append(14999)
        if context!= "":
            if context.lower() in word_count_table:
                table_3.append(word_count_table[context.lower()])
            else:
                table_3.append(14999)
            context = ""

        if len(table_3)!= 667:
            # table_1.append(np.array(table_3[0:-1]))
            # table_2.append(np.array(table_3[1:]))
            print(table_3)
        else:
            table_1.append(np.array(table_3[0:-1]))
            table_2.append(np.array(table_3[1:]))
        if i % 1000 == 0:
            print("Data converted to numpy arrays progress percentage {:.2f}".format(i / len(input_data) * 100))
        i += 1
    print("Data converted to numpy arrays.")

    input_np = np.array(table_1)
    output_np = np.array(table_2)
    np.savez(numpy_array_path, output_np=output_np, input_np=input_np)


def generate_test_numpy_arrays(input_form, word_count_table):
    table_1 = []
    for char in input_form:
        if char.lower() in word_count_table:
            table_1.append(word_count_table[char])
        else:
            table_1.append(14999)
    input_np = np.array(table_1)
    return input_np


def generate_training_arrays_A(input_data, word_count_table, numpy_array_path):
    table_1 = []
    table_2 = []
    i = 0
    context = ""
    for form in input_data:
        table_3 = []
        for char in form:
            if (ord('a') <= ord(char) <= ord('z')) or (ord('A') <= ord(char) <= ord('Z')):
                if context == "":
                    context = char
                else:
                    context += char
            else:
                if context == "":
                    if char.lower() in word_count_table:
                        if char!= " ":
                            table_3.append(word_count_table[char.lower()])
                    else:
                        table_3.append(14999)
                else:
                    if context.lower() in word_count_table:
                        if context!= " ":
                            table_3.append(word_count_table[context.lower()])
                    else:
                        table_3.append(14999)
                    context = ""
                    if char.lower() in word_count_table:
                        if char!= " ":
                            table_3.append(word_count_table[char.lower()])
                    else:
                        table_3.append(14999)
        if context!= "":
            if context.lower() in word_count_table:
                if context!= " ":
                    table_3.append(word_count_table[context.lower()])
            else:
                table_3.append(14999)
            context = ""

        if len(table_3)!= 667:
            # table_1.append(np.array(table_3[0:-1]))
            # table_2.append(np.array(table_3[1:]))
            print(table_3)
        else:
            table_1.append(np.array(table_3[0:-1]))
            table_2.append(np.array(table_3[1:]))
        if i % 1000 == 0:
            print("Data converted to numpy arrays progress percentage {:.2f}".format(i / len(input_data) * 100))
        i += 1
    print("Data converted to numpy arrays.")

    input_np = np.array(table_1)
    output_np = np.array(table_2)
    np.savez(numpy_array_path, output_np=output_np, input_np=input_np)


    def read_training_data_A(path):
        input_forms = []
        with open(path, encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                json_line = json.loads(line)
            
                content = json_line["input"]
                input_forms.append(content)
        return input_forms


def generate_test_numpy_array_A(input_forms, word_count_table):
    table_3 = []
    temp = ""
    for char in input_forms:
        if char.lower() in word_count_table:
            if (ord('A') <= ord(char) <= ord('Z')) or (ord('a') <= ord(char) <= ord('z')):
                if temp == "":
                    temp = char
                else:
                    temp = temp + char
            else:
                if temp == "":
                    if char.lower() in word_count_table:
                        if char.lower()!= " ":
                            table_3.append(word_count_table[char.lower()])
                    else:
                        table_3.append(14999)
                else:
                    if temp.lower() in word_count_table:
                        if temp.lower()!= " ":
                            table_3.append(word_count_table[temp.lower()])
                    else:
                        table_3.append(14999)
                    temp = ""
                    if char.lower() in word_count_table:
                        if char.lower()!= " ":
                            table_3.append(word_count_table[char.lower()])
                    else:
                        table_3.append(14999)
    input_np = np.array(table_3)
    return input_np