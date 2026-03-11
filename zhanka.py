import torch
import torch.multiprocessing as mp

def _str2dtype(s):
    return {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }[s.lower()]

def worker(rank, world_size, target_mem_frac=0.90, mat_dim=8192, dtype_str="float32", batched=False):
    """
    在每张卡上尽量占到 target_mem_frac 的显存，并持续做大矩阵乘。
    - target_mem_frac: 目标显存占用比例（0~1），建议 0.85~0.95，留出 cuBLAS/cudnn workspace。
    - mat_dim: 单个矩阵维度 N（做 N x N 的乘法）。按需增大/减小。
    - dtype_str: 'float32' / 'bfloat16' / 'float16'
    - batched: True 则做批量矩阵乘（B,N,N）·（B,N,N）
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # A100 上建议开启 TF32（float32 下更高吞吐）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    dtype = _str2dtype(dtype_str)
    bytes_per_elem = torch.tensor([], dtype=dtype, device=device).element_size()

    # 查询当前显存
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    # 预估计算时需要的工作集：a、b、c 三个 N×N 矩阵
    def workset_bytes(batch=1):
        return 3 * batch * (mat_dim**2) * bytes_per_elem

    batch = 1
    if batched:
        # 估个 batch，便于一次性做大批量；留一点余量避免 OOM
        batch = max(1, int( (total_bytes * 0.1) // ( (mat_dim**2) * bytes_per_elem )))
        # 太小就干脆退回单大矩阵
        if batch < 8:
            batched = False
            batch = 1

    need_for_compute = workset_bytes(batch)
    # 目标占用 = total * frac；填充显存 = 目标 - 已占用 - 计算工作集
    filler_bytes = int(total_bytes * target_mem_frac) - (total_bytes - free_bytes) - need_for_compute
    filler = None
    if filler_bytes > 0:
        try:
            filler = torch.empty(filler_bytes, dtype=torch.uint8, device=device)
        except RuntimeError:
            # 若一次吃不下，就少吃点
            filler = torch.empty(int(filler_bytes * 0.9), dtype=torch.uint8, device=device)

    # 分配计算张量
    if batched:
        a = torch.randn((batch, mat_dim, mat_dim), device=device, dtype=dtype)
        b = torch.randn((batch, mat_dim, mat_dim), device=device, dtype=dtype)
    else:
        a = torch.randn((mat_dim, mat_dim), device=device, dtype=dtype)
        b = torch.randn((mat_dim, mat_dim), device=device, dtype=dtype)

    print(f"[rank {rank}] device={torch.cuda.get_device_name(rank)}  "
          f"target_mem_frac={target_mem_frac}  mat_dim={mat_dim}  "
          f"dtype={dtype}  batched={batched}  batch={batch}")

    # 无限循环做大矩阵乘
    while True:
        c = a @ b  # batched=False 时是 N×N；batched=True 时是 (B,N,N)·(B,N,N)
        # 避免 Python 回收触发频繁的同步，这里不立刻 del；由下一次覆写 c
        # 偶尔同步一下，避免队列无限膨胀（也便于你在 nvidia-smi 看到稳定的利用率）
        torch.cuda.synchronize(device)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("未检测到CUDA设备，请检查驱动和PyTorch安装。")
    else:
        gpu_count = torch.cuda.device_count()
        mp.set_start_method("spawn", force=True)
        print(f"检测到 {gpu_count} 个GPU，将全部用于计算。")
        # 你可以调 target_mem_frac / mat_dim / dtype_str / batched 来控制显存与算力
        mp.spawn(
            worker,
            args=(gpu_count, 0.9, 8192, "bfloat16", True),  # 改成 ("bfloat16", True) 试试更高吞吐的批量乘
            nprocs=gpu_count,
            join=True
        )
