import os
import shutil
from ctypes import *

import cv2

from utils.config import *


def getP2PtrainData(src_path, res_path):
    for char_dir in os.listdir(src_path):
        abs_char_dir = os.path.join(src_path, char_dir)
        for char_path in os.listdir(abs_char_dir):
            print(char_path)
            abs_char_path = os.path.join(abs_char_dir, char_path)

            char_img = cv2.imread(abs_char_path, cv2.IMREAD_GRAYSCALE)
            bin_img = cv2.adaptiveThreshold(char_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 20)

            char_img = cv2.resize(char_img, (28, 28))
            bin_img = cv2.resize(bin_img, (28, 28))
            res_img = cv2.hconcat([char_img, bin_img])

            cv2.imwrite(os.path.join(res_path, char_path), res_img)


def delete_error_label(train_data_dir):
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    label_dirs = os.listdir(train_data_dir)
    for label_dir in label_dirs:
        train_path = os.path.join(train_data_dir, label_dir)
        img_file_names = os.listdir(train_path)
        for img_file_name in img_file_names:
            if ("_4" in img_file_name) or ("_9" in img_file_name) \
                    or ("_10" in img_file_name) or ("_18" in img_file_name) \
                    or ("_21" in img_file_name) or ("_24" in img_file_name) \
                    or ("_27" in img_file_name) or ("_28" in img_file_name) or ("_32" in img_file_name):
                img_path = os.path.join(train_path, img_file_name)
                print(img_path)
                os.remove(img_path)


def copy_img(path_list, src_dir, dst_dir):
    if isinstance(path_list, list):
        for path_name in path_list:
            shutil.copy(src_dir+path_name, dst_dir+path_name)

    else:
        count = 0
        for img_dir in os.listdir(src_dir):
            for img_file in os.listdir(src_dir + img_dir):
                shutil.copy(src_dir + img_dir + os.sep + img_file, dst_dir + str(count) + ".jpg")
                count += 1


def gen_label():
    with open("chinese_labels_all", "w") as f:
        count = 0
        for code in ASCII_LIST_ALL:
            if count > 0:
                line1 = "p" + str(count) + "\n"
                line2 = "sI" + str(count) + "\n"
                line3 = "V\\u"+code[2:]+ "\n"
                f.write(line1 + line2 + line3)
            else:
                line1 = "(dp" + str(count) + "\n"
                line2 = "I" + str(count) + "\n"
                line3 = "V\\u" + code[2:] + "\n"
                f.write(line1 + line2 + line3)

            count+=1

        f.write("p"+str(count)+"\n"+"s.")


def get_train_char_path(train_dir, char_str):
    char_code = hex(ord(char_str))
    if len(char_code) < 6:
        char_code = char_code[:2] + "0" * (6 - len(char_code)) + char_code[2:]

    # dir_code = str(ASCII_LIST.index(char_code))
    # dir_code = "0" * (5 - len(dir_code)) + dir_code

    char_path = os.path.join(train_dir, char_code)

    if not os.path.exists(char_path):
        os.mkdir(char_path)

    return char_path


def split_word(img_path):
    dll = "SPLIT_WORD_OCR.dll"
    lib = cdll.LoadLibrary(dll)

    b_url = img_path.encode("utf-8")
    c_url = create_string_buffer(b_url)

    lib.test.restype = c_int
    res = lib.test(c_url)

    return res


def comp_str(str1, str2):
    # str1: 标注字符串
    # str2: 模型预测字符串
    len1 = len(str1)
    len2 = len(str2)

    comp_list = []
    for i in range(len1 + 1):
        one_comp_list = [0 for i in range(len2 + 1)]
        comp_list.append(one_comp_list)

    for i in range(1, len1+1):
        for j in range(1, len2+1):
            if str1[i-1] == str2[j-1]:
                comp_list[i][j] = comp_list[i-1][j-1] + 1
            else:
                comp_list[i][j] = max(comp_list[i][j - 1], comp_list[i - 1][j])
    if (len1 + len2) == 0:
        similar = 0
    else:
        similar = 1 - (len1 + len2 -2 * comp_list[len1][len2]) / (len1 + len2)

    return similar

