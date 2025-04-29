import argparse
import numpy as np
import scipy.sparse as sp
import time
import psutil
import os
from scipy.sparse import csr_matrix  # 改为CSR格式
from scipy.sparse.csgraph import connected_components
import threading

TOL = 1e-6
DATA_FILE = "Data.txt"

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

def load_graph(filename):
    """单次IO优化加载图并可选检测蜘蛛节点"""
    # data = load_text_chunked()
    data = np.loadtxt(filename, dtype=np.int32)
    src = data[:, 0]
    dst = data[:, 1]
    max_node = max(src.max(), dst.max())
    n = max_node + 1
    out_degree = np.bincount(src, minlength=n).astype(np.float32) 
    inv_degree = 1.0 / out_degree[src]
    # 使用CSR格式替代CSC格式
    adj = csr_matrix((inv_degree, (dst, src)), shape=(n, n), dtype=np.float32)
    adj.sum_duplicates()
    return adj, (out_degree == 0)


def pagerank(adj, dead_mask, alpha=0.85, tol=TOL, max_iter=50000):
    """标准Pagerank，稀疏矩阵（CSR）直接乘"""
    n = adj.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    pr_next = np.zeros_like(pr)
    teleport = (1 - alpha) / n
    alpha_f32 = np.float32(alpha)
    teleport_f32 = np.float32(teleport)

    dead_nodes = np.where(dead_mask)[0]
    for it in range(max_iter):
        dead_sum = pr[dead_nodes].sum()
        pr_next[:] = adj.dot(pr)
        # pr_next = alpha_f32 * pr_next
        np.multiply(pr_next, alpha_f32, out=pr_next)
        pr_next += alpha_f32 * dead_sum / n + teleport_f32

        if np.abs(pr_next - pr).sum() < tol:
            print(f"Converged at {it+1} iterations, delta={np.abs(pr_next-pr).sum():.2e}")
            break
        pr, pr_next = pr_next, pr
    return pr


def block_pagerank(adj, dead_mask, alpha=0.85, tol=TOL, max_iter=50000, block_size=None):
    """Block Matrix版PageRank（分块大小就是数据集大小，CSR版）"""
    n = adj.shape[0]
    if block_size is None or block_size >= n:
        block_size = n
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    teleport = (1 - alpha) / n
    alpha_f32 = np.float32(alpha)
    teleport_f32 = np.float32(teleport)
    dead_nodes = np.where(dead_mask)[0]

    for it in range(max_iter):
        dead_sum = pr[dead_nodes].sum()
        pr_new = np.zeros_like(pr)
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = adj[start:end, :]
            pr_new[start:end] = block.dot(pr)
        np.multiply(pr_new, alpha_f32, out=pr_new)
        pr_new += alpha_f32 * dead_sum / n + teleport_f32
        if np.abs(pr_new - pr).sum() < tol:
            print(f"[Block] Converged at {it+1} iterations, delta={np.abs(pr_new-pr).sum():.2e}")
            break
        pr = pr_new
    return pr


def save_topk(pr, topk=100, filename="Res.txt"):
    """缓存优化的Top-K选择"""
    top_indices = np.argpartition(-pr, topk)[:topk]
    top_indices = top_indices[np.argsort(-pr[top_indices])]
    with open(filename, "w") as f:
        buffer = []
        for idx in top_indices:
            buffer.append(f"{idx} {pr[idx]:.10f}\n")
            if len(buffer) >= 1000:
                f.writelines(buffer)
                buffer = []
        if buffer:
            f.writelines(buffer)


class MemoryMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.running = True

    def run(self):
        proc = psutil.Process(os.getpid())
        while self.running:
            self.peak = max(self.peak, proc.memory_info().rss)
            time.sleep(0.01)

    def stop(self):
        self.running = False


def main():
    parser = argparse.ArgumentParser(description="PageRank Algorithm")
    parser.add_argument("--input", default="Data.txt", help="Input file path")
    parser.add_argument("--output", default="Res.txt", help="Output file path")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence threshold")
    parser.add_argument("--use_block", default=0, help="Use block matrix 0/1")
    args = parser.parse_args()

    global DATA_FILE, TOL
    DATA_FILE = args.input
    TOL = args.tol

    print("运行main.py")
    monitor = MemoryMonitor()
    monitor.start()

    try:
        t_start = time.time()
        adj, dead_mask = load_graph(args.input)
        n = adj.shape[0]
        threshold = max(1000, int(n * 0.05))
        use_block = args.use_block == 1
        t_calc = time.time()
        if use_block:
            pr = block_pagerank(adj, dead_mask)
        else:
            pr = pagerank(adj, dead_mask)
        print(f"PageRank耗时: {time.time()-t_calc:.3f}s")
        save_topk(pr, filename=args.output)
    finally:
        monitor.stop()
        monitor.join()
    print(f"峰值内存: {monitor.peak/(1024 * 1024):.2f}MB")
    print(f"总耗时: {time.time()-t_start:.3f}s")


if __name__ == "__main__":
    main()
