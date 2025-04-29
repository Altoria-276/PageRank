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
import threading

# ================== 实验配置 ==================
DATA_FILE = "Data.txt"
TOL = 1e-6  # 收敛容忍度
MEM_SAMPLE_INTERVAL = 0.001  # 内存采样间隔1ms


# ================== 增强版性能监控装饰器 ==================
class MemoryMonitor:
    """内存峰值监控器"""

    def __init__(self):
        self.peak = 0
        self._stop = False

    def monitor(self):
        proc = psutil.Process(os.getpid())
        self.peak = proc.memory_info().rss
        while not self._stop:
            self.peak = max(self.peak, proc.memory_info().rss)
            time.sleep(MEM_SAMPLE_INTERVAL)


def experiment(name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            gc.collect()
            base_mem = psutil.Process().memory_info().rss
            monitor = MemoryMonitor()
            t = threading.Thread(target=monitor.monitor)
            t.start()
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                monitor._stop = True
                t.join()
            return {"name": name, "time": time.time() - start_time, "memory": (monitor.peak - base_mem) / (1024**2), "result": result}

        return wrapper

    return decorator


# ================== 数据加载方法 ==================
@experiment("1.一次性加载(baseline)")
def load_text_full():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return np.loadtxt(f, dtype=np.uint32)


@experiment("2.分块加载(1万行/块)")
def load_text_chunked():
    chunks = []
    chunk_size = 10000
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        while True:
            chunk = np.loadtxt(f, dtype=np.uint32, max_rows=chunk_size)
            if chunk.size == 0:
                break
            chunks.append(chunk)
    return np.concatenate(chunks) if chunks else np.empty((0, 2), dtype=np.uint32)


@experiment("3.内存映射加载(基于mmap+np.fromstring)")
def load_text_memmap():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        data_str = mm.read().decode("ascii")
        mm.close()
        arr = np.fromstring(data_str, dtype=np.uint32, sep=" ")
        return arr.reshape(-1, 2)


# ================== 矩阵构建方法 ==================
def validate_data(data):
    if data.size == 0:
        raise ValueError("数据文件为空")
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("数据格式不正确，应为N×2的边列表")
    return data


@experiment("1.CSC优化构建")
def build_csc_optimized(data):
    data = validate_data(data)
    src, dst = data[:, 0], data[:, 1]
    n = max(src.max(), dst.max()) + 1
    out_degree = np.bincount(src, minlength=n).astype(np.float32)
    np.maximum(out_degree, 1.0, out=out_degree)
    inv_degree = 1.0 / out_degree[src]
    return csc_matrix((inv_degree, (dst, src)), shape=(n, n))


@experiment("2.CSC优化+矩阵预分配构建")
def build_csc_matrix_optimized(data):
    data = validate_data(data)
    src, dst = data[:, 0], data[:, 1]
    n = max(src.max(), dst.max()) + 1
    out_degree = np.bincount(src, minlength=n).astype(np.float32)
    np.maximum(out_degree, 1.0, out=out_degree)
    inv_degree = 1.0 / out_degree[src]
    nnz = len(src)
    indices = np.empty(nnz, dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)
    data_vals = np.empty(nnz, dtype=np.float32)
    np.cumsum(np.bincount(dst, minlength=n), out=indptr[1:])
    pos = indptr[:-1].copy()
    for i in range(nnz):
        d = dst[i]
        idx = pos[d]
        indices[idx] = src[i]
        data_vals[idx] = inv_degree[i]
        pos[d] += 1
    return csc_matrix((data_vals, indices, indptr), shape=(n, n))


# ================== PageRank方法 ==================
@experiment("1.基础PageRank(baseline)")
def pagerank_baseline(adj, alpha=0.85, max_iter=100):
    n = adj.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    teleport = (1 - alpha) / n
    adj_alpha = alpha * adj
    for i in range(max_iter):
        old_pr = pr.copy()
        pr = adj_alpha.dot(pr) + teleport
        if np.abs(pr.sum() - 1.0) > 1e-6:
            pr /= pr.sum()
        if np.linalg.norm(pr - old_pr, 1) < TOL:
            break
    return {"iterations": i + 1, "pr": pr}


@experiment("2.增强型PageRank(含死结点处理)")
def pagerank_enhanced(adj, alpha=0.85, max_iter=100):
    n = adj.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    out_degree = adj.sum(axis=0).A1
    dangling = out_degree == 0
    teleport = (1 - alpha) / n
    for i in range(max_iter):
        old_pr = pr.copy()
        dangling_sum = alpha * old_pr[dangling].sum()
        pr = alpha * adj.dot(old_pr) + teleport + dangling_sum / n
        pr /= pr.sum()
        if np.linalg.norm(pr - old_pr, 1) < TOL:
            break
    return {"iterations": i + 1, "pr": pr}


@experiment("3.分块迭代优化")
def pagerank_block(adj, alpha=0.85, block_size=1000, max_iter=100):
    n = adj.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    teleport = (1 - alpha) / n
    blocks = [slice(i, min(i + block_size, n)) for i in range(0, n, block_size)]
    for i in range(max_iter):
        old_pr = pr.copy()
        for blk in blocks:
            pr[blk] = alpha * adj[blk, :].dot(old_pr) + teleport
        pr /= pr.sum()
        if np.linalg.norm(pr - old_pr, 1) < TOL:
            break
    return {"iterations": i + 1, "pr": pr}


@experiment("4.分块矩阵迭代优化(基于子矩阵)")
def pagerank_block_matrix(adj, alpha=0.85, block_size=1000, max_iter=100):
    n = adj.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    teleport = (1 - alpha) / n
    idxs = list(range(0, n, block_size)) + [n]
    blocks = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    for it in range(max_iter):
        old_pr = pr.copy()
        new_pr = np.zeros_like(pr)
        for i_start, i_end in blocks:
            for j_start, j_end in blocks:
                sub = adj[i_start:i_end, j_start:j_end]
                new_pr[i_start:i_end] += alpha * sub.dot(old_pr[j_start:j_end])
        new_pr += teleport
        new_pr /= new_pr.sum()
        pr = new_pr
        if np.linalg.norm(pr - old_pr, 1) < TOL:
            break
    return {"iterations": it + 1, "pr": pr}


@experiment("5.蜘蛛陷阱检测和处理")
def pagerank_spider_trap(adj, alpha=0.85, max_iter=100):
    n = adj.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    teleport = (1 - alpha) / n
    out_degree = adj.sum(axis=0).A1
    dangling = out_degree == 0
    n_comp, labels = connected_components(adj, directed=True, connection="strong")
    trap_mask = np.zeros(n, dtype=bool)
    for comp in range(n_comp):
        nodes = np.where(labels == comp)[0]
        sub = adj[nodes, :][:, nodes]
        if sub.sum() == adj[nodes, :].sum():
            trap_mask[nodes] = True
    for i in range(max_iter):
        old = pr.copy()
        dangling_sum = alpha * old[dangling].sum()
        trap_sum = alpha * old[trap_mask].sum()
        pr = alpha * adj.dot(old) + teleport + dangling_sum / n + trap_sum / n
        pr /= pr.sum()
        if np.linalg.norm(pr - old, 1) < TOL:
            break
    return {"iterations": i + 1, "pr": pr}


# ================== 主流程 ==================
def run_experiments():
    def pad(s, width=40):
        count = 0
        for ch in s:
            if "\u4e00" <= ch <= "\u9fff":
                count += 2  # 中文宽度为2
            else:
                count += 1  # 英文宽度为1
        return s + " " * (width - count)

    def print_table(results, title, show_iters=False):

        print(f"\n{title:-^60}")
        header = f"{pad('方法',40)} | {pad('时间(s)',8)} | {pad('内存(MB)',9)}"
        if show_iters:
            header += " | 迭代次数"
        print(header)
        print("-" * 60)
        for res in results:
            line = f"{pad(res['name'],40)} | {pad(f'{res['time']:>8.3f}',8)} | {pad(f'{res['memory']:>9.2f}',9)}"
            if show_iters:
                line += f" | {res.get('iterations', ''):>8}"
            print(line)

    # 数据加载测试
    print("\n=== 数据加载测试 ===")
    io_results = []
    for loader in [load_text_full, load_text_chunked, load_text_memmap]:
        result = loader()
        io_results.append({"name": result["name"], "time": result["time"], "memory": result["memory"]})
        if "result" in result:
            del result["result"]
        gc.collect()

    # 矩阵构建测试
    print("\n=== 矩阵构建测试 ===")
    matrix_results = []
    data = None
    try:
        data_result = load_text_full()
        data = data_result["result"]
        del data_result
        for builder in [build_csc_optimized, build_csc_matrix_optimized]:
            result = builder(data)
            matrix_results.append({"name": result["name"], "time": result["time"], "memory": result["memory"]})
            if "result" in result:
                del result["result"]
            gc.collect()
    except Exception as e:
        print(f"矩阵构建失败: {str(e)}")
    finally:
        if data is not None:
            del data
        gc.collect()

    # PageRank测试
    print("\n=== PageRank算法测试 ===")
    pr_results = []
    adj = None
    try:
        data_result = load_text_full()
        data = data_result["result"]
        del data_result
        builder_result = build_csc_optimized(data)
        adj = builder_result["result"]
        del builder_result
        del data
        for pagerank in [pagerank_baseline, pagerank_enhanced, pagerank_block, pagerank_block_matrix, pagerank_spider_trap]:
            result = pagerank(adj)
            pr_results.append(
                {"name": result["name"], "time": result["time"], "memory": result["memory"], "iterations": result["result"]["iterations"]}
            )
            if "result" in result:
                del result["result"]
            gc.collect()
    except Exception as e:
        print(f"PageRank失败: {str(e)}")
    finally:
        if adj is not None:
            del adj
        gc.collect()

    print_table(io_results, " 数据加载性能 ")
    print_table(matrix_results, " 矩阵构建性能 ")
    print_table(pr_results, " PageRank性能对比 ", show_iters=True)


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"错误：未找到数据文件 {DATA_FILE}")
    else:
        run_experiments()
    # print("\n最终内存占用：", f"{psutil.Process().memory_info().rss / 1024**2:.2f} MB")
