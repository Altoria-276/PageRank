import numpy as np
import scipy.sparse as sp
import time
import psutil
import os
import gc
import mmap
from functools import wraps
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
import threading 
import warnings 
import multiprocessing as mp 

warnings.filterwarnings("ignore", category=UserWarning)  

# ================== 实验配置 ==================
DATA_FILE = "Data.txt"
TOL = 1e-6
MEM_SAMPLE_INTERVAL = 0.000001

# ================== 内存监控装饰器 ==================
class MemoryMonitor:
    def __init__(self):
        self.peak = 0
        self._stop = False
    
    def monitor(self, pid):
        proc = psutil.Process(pid)
        self.peak = proc.memory_info().rss
        while not self._stop:
            current_mem = proc.memory_info().rss
            if current_mem > self.peak:
                self.peak = current_mem
            time.sleep(MEM_SAMPLE_INTERVAL)

def experiment(func): 
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        proc = psutil.Process(os.getpid())
        base_mem = proc.memory_info().rss  # 正确获取初始内存
        monitor = MemoryMonitor()
        monitor_thread = threading.Thread(target=monitor.monitor, args=(os.getpid(),))
        monitor_thread.start()
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally: 
            monitor._stop = True
            monitor_thread.join()
        memory_usage = (monitor.peak - base_mem) / (1024 ** 2)  # 计算内存增量
        return {
            "name": func.__name__,
            "time": time.time() - start_time,
            "memory": memory_usage,
            "result": result
        }
    return wrapper


# ================== 数据加载方法 ==================
@experiment
def load_full():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return np.loadtxt(f, dtype=np.uint32)

@experiment
def load_chunked():
    chunks, size = [], 1000
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        while True:
            chunk = np.loadtxt(f, dtype=np.uint32, max_rows=size)
            if chunk.size == 0:
                break
            chunks.append(chunk)
    return np.concatenate(chunks) if chunks else np.empty((0,2), np.uint32)

@experiment
def load_memmap():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        arr = np.fromstring(mm.read().decode('ascii'), dtype=np.uint32, sep=' ')
        mm.close()
        return arr.reshape(-1,2)

# ================== 矩阵构建方法 ==================
def validate(data):
    if data.size==0 or data.ndim!=2 or data.shape[1]!=2:
        raise ValueError("边列表数据不合法")
    return data

@experiment
def build_csc(data):
    data = validate(data)
    src, dst = data[:,0], data[:,1]
    n = max(src.max(), dst.max())+1
    deg = np.bincount(src, minlength=n).astype(np.float32)
    np.maximum(deg, 1.0, out=deg)
    vals = 1.0/deg[src]
    return csc_matrix((vals, (dst, src)), shape=(n,n))

@experiment
def build_csr(data):
    data = validate(data)
    src, dst = data[:,0], data[:,1]
    n = max(src.max(), dst.max())+1
    deg = np.bincount(src, minlength=n).astype(np.float32)
    np.maximum(deg, 1.0, out=deg)
    vals = 1.0/deg[src]
    return csr_matrix((vals, (src, dst)), shape=(n,n))

@experiment
def build_coo(data):
    data = validate(data)
    src, dst = data[:,0], data[:,1]
    n = max(src.max(), dst.max())+1
    mat = coo_matrix((np.ones_like(src, np.float32),(dst,src)), shape=(n,n))
    deg = np.bincount(src, minlength=n).astype(np.float32)
    np.maximum(deg,1.0,out=deg)
    mat.data = 1.0/deg[mat.col]
    return mat.tocsc()

@experiment
def build_block(data, block_size=10000):
    # 可替换为实际分块构建逻辑，此处仍使用CSC
    return build_csc(data)["result"]

# ================== PageRank方法 ==================
@experiment
def pr_csc(adj, alpha=0.85, max_iter=100):
    mat = adj # .tocsc()
    n = mat.shape[0]
    pr = np.full(n,1.0/n,dtype=np.float32)
    tp = (1-alpha)/n
    outd = mat.sum(axis=1).A1==0
    for i in range(max_iter):
        old = pr.copy()
        ds = alpha * old[outd].sum()
        pr = alpha*mat.dot(old) + tp + ds/n 
        pr /= pr.sum()
        if np.linalg.norm(pr-old,1)<TOL: break
    return {"iters":i+1, "pr":pr}

@experiment
def pr_csr(adj, alpha=0.85, max_iter=100):
    mat = adj # .tocsr()
    n = mat.shape[0]
    pr = np.full(n,1.0/n,dtype=np.float32)
    tp = (1-alpha)/n 
    outd = mat.sum(axis=1).A1==0
    for i in range(max_iter):
        old = pr.copy() 
        ds = alpha * old[outd].sum()
        pr = alpha * mat.dot(old) + tp + ds/n
        pr /= pr.sum()
        if np.linalg.norm(pr-old,1)<TOL: break
    return {"iters":i+1, "pr":pr}

@experiment
def pr_coo(adj, alpha=0.85, max_iter=100):
    mat = adj # .tocsc()
    n = mat.shape[0]
    pr = np.full(n,1.0/n,dtype=np.float32)
    tp = (1-alpha)/n 
    outd = mat.sum(axis=1).A1==0 
    for i in range(max_iter):
        old = pr.copy() 
        ds = alpha * old[outd].sum()
        pr = alpha * mat.dot(old) + tp +ds/n 
        pr /= pr.sum()
        if np.linalg.norm(pr-old,1)<TOL: break
    return {"iters":i+1, "pr":pr}

@experiment
def pr_block(adj, alpha=0.85, block_size=1000, max_iter=100):
    n = adj.shape[0]
    pr = np.full(n,1.0/n,dtype=np.float32)
    tp = (1-alpha)/n
    idxs = list(range(0,n,block_size)) + [n]
    blks = [(idxs[i], idxs[i+1]) for i in range(len(idxs)-1)]
    for i in range(max_iter):
        old = pr.copy()
        new = np.zeros_like(pr)
        for a,b in blks:
            new[a:b] += alpha * adj[a:b, :].dot(old)
        pr = (new + tp) / (new.sum() + tp*n)
        if np.linalg.norm(pr-old,1)<TOL: break
    return {"iters":i+1, "pr":pr}

# ================== 运行实验（多进程隔离） ==================
def run_combined_experiment(builder, pr_method, loader, queue):
    @experiment
    def combined_experiment():
        data_result = loader()
        data = data_result['result']
        build_res = builder(data)
        adj = build_res['result']
        pr_res = pr_method(adj)
        del data, build_res, adj
        gc.collect()
        return pr_res['result']
    result = combined_experiment()
    queue.put({
        "name": f"{builder.__name__}+{pr_method.__name__}",
        "time": result['time'],
        "memory": result['memory'],
        "iters": result['result']['iters']
    })

def run_experiments():
    loaders = [load_full, load_chunked, load_memmap]
    builders = [build_csc, build_csr, build_coo, build_block]
    pr_methods = [pr_csc, pr_csr, pr_coo, pr_block]

    # 数据加载测试
    print("--- 数据加载性能 ---")
    for loader in loaders:
        proc = mp.Process(target=loader_wrapper, args=(loader,))
        proc.start()
        proc.join()

    # 组合实验测试
    print("\n--- 构建 + PageRank 组合实验 ---")
    for builder, pr_method in zip(builders, pr_methods):
        queue = mp.Queue()
        proc = mp.Process(
            target=run_combined_experiment,
            args=(builder, pr_method, load_full, queue)
        )
        proc.start()
        proc.join()
        result = queue.get()
        print(f"{result['name']}: 时间 {result['time']:.3f}s, 内存 {result['memory']:.2f}MB, 迭代 {result['iters']}")

def loader_wrapper(loader):
    result = loader()
    print(f"{result['name']}: 时间 {result['time']:.3f}s, 内存 {result['memory']:.2f}MB")

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"未找到 {DATA_FILE}")
    else:
        run_experiments() 
