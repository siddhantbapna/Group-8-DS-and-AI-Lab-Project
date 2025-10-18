
import torch, platform, subprocess, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch.version.cuda:", torch.version.cuda)  # None => CPU-only build
print("cuDNN enabled:", torch.backends.cudnn.enabled)
print("num devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))

# Try querying NVIDIA driver
try:
    out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
    print("\n=== nvidia-smi ===\n", out[:500])
except Exception as e:
    print("nvidia-smi not available:", e)

print("python:", sys.version)
print("os:", platform.platform())