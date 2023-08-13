#!/bin/bash

pycode="
import ctypes

dll = ctypes.cdll.LoadLibrary('libcuda.so')
major = ctypes.c_int(0)
minor = ctypes.c_int(0)
dll.cuInit(ctypes.c_int(0))
device = 0
ret = dll.cuDeviceComputeCapability(ctypes.pointer(major), ctypes.pointer(minor), ctypes.c_int(device))
ret = int(ret)
if ret != 0:
    exit(ret)

name = ctypes.create_string_buffer(100)
ret = dll.cuDeviceGetName(ctypes.pointer(name), 100, device)

name = str(name.value, encoding='utf-8')
major = major.value
minor = minor.value

sm = f'{major}{minor}'
print(sm, end='')
"

cudasm=`python3 -c "$pycode"`