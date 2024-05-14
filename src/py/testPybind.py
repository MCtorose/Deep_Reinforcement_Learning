import os
import sys
os.add_dll_directory(os.environ['mingwPath'])
pybind_path = os.path.join(r"E:\\Desktop\\drl\\build")
sys.path.insert(0, pybind_path)

import pybindDemo

print(pybindDemo.add(1, 2))
