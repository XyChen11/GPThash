import torch
print("CUDA 可用:", torch.cuda.is_available())
print("GPU 数量:", torch.cuda.device_count())
print("当前设备:", torch.cuda.current_device())
print("设备名称:", torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "无 GPU")