"""
CUDA and Triton diagnostic script
Run: python cuda_triton_diagnostic.py
"""

import sys

print("=" * 60)
print("CUDA & Triton Diagnostic")
print("=" * 60)

# Python Version
print(f"\n[Python]")
print(f"  Version: {sys.version}")

# CUDA Check via PyTorch
print(f"\n[PyTorch & CUDA]")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    - Compute capability: {props.major}.{props.minor}")
            print(f"    - Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"    - Multiprocessors: {props.multi_processor_count}")
    else:
        print("  ⚠ CUDA not available")
except ImportError:
    print("  ✗ PyTorch not installed")

# Triton Check
print(f"\n[Triton]")
try:
    import triton
    print(f"  Triton version: {triton.__version__}")
    
    import triton.language as tl
    
    @triton.jit
    def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)
    
    print(f"  Triton JIT: ✓")
except ImportError as e:
    print(f"  ✗ Triton not installed: {e}")
except Exception as e:
    print(f"  ⚠ Triton error: {e}")

# torch.compile Check
print(f"\n[torch.compile]")
try:
    import torch
    if hasattr(torch, 'compile'):
        print(f"  Available: ✓")
        if torch.cuda.is_available():
            model = torch.nn.Linear(64, 64).cuda()
            compiled = torch.compile(model)
            _ = compiled(torch.randn(32, 64).cuda())
            print(f"  Test (CUDA): ✓")
    else:
        print(f"  Available: ✗ (need PyTorch 2.0+)")
except Exception as e:
    print(f"  Test failed: {e}")

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)

try:
    import torch
    cuda_ok = torch.cuda.is_available()
except:
    cuda_ok = False

try:
    import triton
    triton_ok = True
except:
    triton_ok = False

if cuda_ok and triton_ok:
    print("✓ Ready for GPU-accelerated RL with Triton!")
elif cuda_ok:
    print("⚠ CUDA works, but Triton missing. Install: pip install triton")
else:
    print("✗ CUDA not available")

print()