import torch
 
if (torch.cuda.is_available()):
    print(f"CUDA is available!")
    print(f"CUDA version: {torch.version.cuda}")
    
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
else:
    print(f"CUDA is not available... Check if you have installed torch with CUDA enabled and the necessary nvidia drivers")