from ctypes import *
from numpy.ctypeslib import ndpointer

from utils.config import ASCII_LIST

if __name__ == '__main__':
    lib = cdll.LoadLibrary("CR_OCR.dll")

    img_url = "13.jpg"
    b_url = img_url.encode('utf-8')
    c_url = create_string_buffer(b_url)

    lib.getLen.restype = c_int
    a = lib.getLen(c_url)

    lib.getWord.restype = ndpointer(dtype=c_int, shape=(a,))
    b = lib.getWord(c_url)

    for c in b:
        print(chr(int(ASCII_LIST[int(c)], 16)), end="")
