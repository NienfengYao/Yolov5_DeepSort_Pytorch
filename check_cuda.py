import cv2, torch


def check_cv2():
	print("Check CV2:", cv2.__version__)
	count = cv2.cuda.getCudaEnabledDeviceCount()
	if count > 0:
		print('\tusing cuda')
	else:
		print('\tnot using cuda')


def check_torch():
	print("Check Torch:", torch.__version__)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('\tUsing device:', device)
	#Additional Info when using cuda
	if device.type == 'cuda':
		print('\t%s' % torch.cuda.get_device_name(0))
		print('\tMemory Usage:')
		print('\tAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
		print('\tCached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


if __name__ == '__main__':
	check_cv2()
	check_torch()
