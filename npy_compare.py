import numpy as np
import os

def compare(f_name):
	path1 = os.path.join('/home/ryanyao/tftpboot_ryanyao/mot/Yolov5_DeepSort_Pytorch', f_name)
	path2 = os.path.join('/home/ryanyao/ros2_dev_ws_develop/src/ros2_deep_sort/deep_sort', f_name)
	print(path1, path2)
	a = np.load(path1, allow_pickle=True)
	b = np.load(path2, allow_pickle=True)
	print(f_name, a.dtype, b.dtype, np.array_equal(a, b))


if __name__ == '__main__':
	compare('imgA_0.npy')
	compare('imgB_0.npy')
	compare('imgC_0.npy')
	compare('imgD_0.npy')
	compare('imgE_0.npy')
	compare('imgF_0.npy')
