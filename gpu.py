import torch, os
print("PID:", os.getpid())
print("CUDA visible devices:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("Currently using GPU id:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # export CUDA_VISIBLE_DEVICES=4,5