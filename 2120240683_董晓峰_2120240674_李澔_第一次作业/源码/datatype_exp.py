import numpy as np
import scipy.sparse as sp
import time
import psutil
import os
import gc
import mmap
from functools import wraps
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.csgraph import connected_components

# ================== 实验配置 ==================
DATA_FILE = "Data.txt"
TOL = 1e-6  # 收敛容忍度


# ================== 性能监控装饰器 ==================
def experiment(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        gc.collect()
        proc = psutil.Process(os.getpid())
        start_mem = proc.memory_info().rss
        start_time = time.time()
        result = fn(*args, **kwargs)
        gc.collect()
        end_mem = proc.memory_info().rss
        return {
            "name": fn.__name__,
            "time": time.time() - start_time,
            "memory": max(0, (end_mem - start_mem) / (1024**2)),
            "result": result,
        }

    return wrapper


# ================== 数据加载方法 ==================
@experiment
def load_full():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return np.loadtxt(f, dtype=np.uint32)


@experiment
def load_chunked():
    chunks, size = [], 10000
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        while True:
            chunk = np.loadtxt(f, dtype=np.uint32, max_rows=size)
            if chunk.size == 0:
                break
            chunks.append(chunk)
    return np.concatenate(chunks) if chunks else np.empty((0, 2), np.uint32)


@experiment
def load_memmap():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        arr = np.fromstring(mm.read().decode("ascii"), dtype=np.uint32, sep=" ")
        mm.close()
        return arr.reshape(-1, 2)


# ================== 矩阵构建方法 ==================
def validate(data):
    if data.size == 0 or data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("边列表数据不合法")
    return data


@experiment
def build_csc(data):
    data = validate(data)
    src, dst = data[:, 0], data[:, 1]
    n = max(src.max(), dst.max()) + 1
    deg = np.bincount(src, minlength=n).astype(np.float32)
    np.maximum(deg, 1.0, out=deg)
    vals = 1.0 / deg[src]
    return csc_matrix((vals, (dst, src)), shape=(n, n))


@experiment
def build_csr(data):
    data = validate(data)
    src, dst = data[:, 0], data[:, 1]
    n = max(src.max(), dst.max()) + 1
    deg = np.bincount(src, minlength=n).astype(np.float32)
    np.maximum(deg, 1.0, out=deg)
    vals = 1.0 / deg[src]
    return csr_matrix((vals, (src, dst)), shape=(n, n))


@experiment
def build_coo(data):
    data = validate(data)
    src, dst = data[:, 0], data[:, 1]
    n = max(src.max(), dst.max()) + 1
    mat = coo_matrix((np.ones_like(src, np.float32), (dst, src)), shape=(n, n))
    deg = np.bincount(src, minlength=n).astype(np.float32)
    np.maximum(deg, 1.0, out=deg)
    mat.data = 1.0 / deg[mat.col]
    return mat.tocsc()


@experiment
def build_block(data, block_size=10000):
    # 可替换为实际分块构建逻辑，此处仍使用CSC
    return build_csc(data)["result"]


# ================== PageRank方法 ==================
@experiment
def pr_csc(adj, alpha=0.85, max_iter=100):
    mat = adj.tocsc()
    n = mat.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    tp = (1 - alpha) / n
    for i in range(max_iter):
        old = pr.copy()
        pr = alpha * mat.dot(old) + tp
        pr /= pr.sum()
        if np.linalg.norm(pr - old, 1) < TOL:
            break
    return {"iters": i + 1, "pr": pr}


@experiment
def pr_csr(adj, alpha=0.85, max_iter=100):
    mat = adj.tocsr()
    n = mat.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    tp = (1 - alpha) / n
    outd = mat.sum(axis=1).A1 == 0
    for i in range(max_iter):
        old = pr.copy()
        ds = alpha * old[outd].sum()
        pr = alpha * mat.dot(old) + tp + ds / n
        pr /= pr.sum()
        if np.linalg.norm(pr - old, 1) < TOL:
            break
    return {"iters": i + 1, "pr": pr}


@experiment
def pr_coo(adj, alpha=0.85, max_iter=100):
    mat = adj.tocsc()
    n = mat.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    tp = (1 - alpha) / n
    for i in range(max_iter):
        old = pr.copy()
        pr = alpha * mat.dot(old) + tp
        pr /= pr.sum()
        if np.linalg.norm(pr - old, 1) < TOL:
            break
    return {"iters": i + 1, "pr": pr}


@experiment
def pr_block(adj, alpha=0.85, block_size=1000, max_iter=100):
    n = adj.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    tp = (1 - alpha) / n
    idxs = list(range(0, n, block_size)) + [n]
    blks = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    for i in range(max_iter):
        old = pr.copy()
        new = np.zeros_like(pr)
        for a, b in blks:
            new[a:b] += alpha * adj[a:b, :].dot(old)
        pr = (new + tp) / (new.sum() + tp * n)
        if np.linalg.norm(pr - old, 1) < TOL:
            break
    return {"iters": i + 1, "pr": pr}


# ================== 实验运行 ==================


def run_experiments():
    loaders = [load_full, load_chunked, load_memmap]
    builders = [build_csc, build_csr, build_coo, build_block]
    pr_methods = [pr_csc, pr_csr, pr_coo, pr_block]

    # 1. 数据加载
    load_results = [f() for f in loaders]
    print("\n--- 数据加载性能 ---")
    for r in load_results:
        print(f"{r['name']}: 时间 {r['time']:.3f}s, 内存 {r['memory']:.2f}MB")

    # 2. 矩阵构建
    data = load_results[0]["result"]
    build_results = [f(data) for f in builders]
    print("\n--- 矩阵构建性能 ---")
    for r in build_results:
        mat = r["result"]
        shape = mat.shape if hasattr(mat, "shape") else "未知"
        print(f"{r['name']}: 时间 {r['time']:.3f}s, 内存 {r['memory']:.2f}MB, 维度 {shape}")

    # 3. 构建+PageRank 组合对比
    print("\n--- 构建 + PageRank 组合实验（累计时间/内存） ---")
    for b_res, b_fn in zip(build_results, builders):
        adj = b_res["result"]
        build_time = b_res["time"]
        build_mem = b_res["memory"]
        for pr_fn in pr_methods:
            pr_res = pr_fn(adj)
            total_time = build_time + pr_res["time"]
            total_mem = build_mem + pr_res["memory"]
            iters = pr_res["result"]["iters"]
            print(f"{b_fn.__name__}+{pr_fn.__name__}: 总时间 {total_time:.3f}s, 总内存 {total_mem:.2f}MB, 迭代 {iters}")


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"未找到 {DATA_FILE}")
    else:
        run_experiments()
