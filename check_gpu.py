import torch
import sys

print("="*60)
print("GPU 環境檢查")
print("="*60)

# 檢查 PyTorch 版本
print(f"\n📦 PyTorch 版本: {torch.__version__}")
print(f"📦 CUDA 版本: {torch.version.cuda}")

# 檢查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"\n🎮 CUDA 可用: {cuda_available}")

if cuda_available:
    # GPU 資訊
    gpu_count = torch.cuda.device_count()
    print(f"🎮 GPU 數量: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\n🎮 GPU {i}:")
        print(f"   名稱: {torch.cuda.get_device_name(i)}")
        print(f"   記憶體總量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 檢查當前記憶體使用
        if hasattr(torch.cuda, 'mem_get_info'):
            free, total = torch.cuda.mem_get_info(i)
            print(f"   可用記憶體: {free / 1024**3:.2f} GB")
            print(f"   已用記憶體: {(total - free) / 1024**3:.2f} GB")
    
    # 測試簡單運算
    print(f"\n🧪 測試 GPU 運算...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✓ GPU 運算測試成功！")
        print(f"✓ 當前使用的 GPU: {torch.cuda.current_device()}")
    except Exception as e:
        print(f"❌ GPU 運算測試失敗: {e}")
else:
    print("\n❌ CUDA 不可用，將使用 CPU 訓練")
    print("\n請檢查:")
    print("1. 是否安裝了支援 CUDA 的 PyTorch")
    print("2. NVIDIA 驅動是否正確安裝")
    print("3. CUDA toolkit 是否安裝")

print("\n" + "="*60)