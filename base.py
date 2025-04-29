import numpy as np
import time
import psutil
import os
import threading

TOL = 1e-6
DATA_FILE = "Data.txt"

def load_graph(filename, use_spider_check=False, spider_threshold=1):
    """基础IO加载图，不使用CSR或快速读取"""
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, v = map(int, parts[:2])
                edges.append((u, v))

    if not edges:
        return None, None, None

    src = [u for u, _ in edges]
    dst = [v for _, v in edges]
    max_node = max(max(src), max(dst))
    n = max_node + 1

    # 计算出度
    out_degree = np.zeros(n, dtype=np.float32)
    for u in src:
        out_degree[u] += 1
    # 避免除零
    out_degree[out_degree == 0] = 1.0

    # 构建密集邻接矩阵，adj[v, u] = 1/out_degree[u]
    adj = np.zeros((n, n), dtype=np.float32)
    for u, v in edges:
        adj[v, u] += 1.0 / out_degree[u]

    # 死节点掩码
    dead_mask = (out_degree == 1.0)

    spider_nodes = None
    if use_spider_check:
        spider_nodes = [i for i, d in enumerate(out_degree) if d <= spider_threshold]
        print(f"Found {len(spider_nodes)} potential spider nodes (out_degree <= {spider_threshold})")

    return adj, dead_mask, spider_nodes

def pagerank(adj, dead_mask, alpha=0.85, tol=TOL, max_iter=50000):
    """标准PageRank，使用密集矩阵乘法"""
    n = adj.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    teleport = (1 - alpha) / n

    for it in range(max_iter):
        old = pr.copy()
        dead_sum = alpha * old[dead_mask].sum()

        pr = alpha * adj.dot(old)
        pr += teleport + dead_sum / n

        if np.abs(pr - old).sum() < tol:
            print(f"Converged at {it+1} iterations, delta={np.abs(pr-old).sum():.2e}")
            break
    return pr

def save_topk(pr, topk=100, filename='Res.txt'):
    """简单Top-K输出，无缓冲优化"""
    top_indices = np.argsort(-pr)[:topk]
    with open(filename, 'w') as f:
        for idx in top_indices:
            f.write(f"{idx} {pr[idx]:.10f}\n")

class MemoryMonitor(threading.Thread):
    """内存监控线程"""
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
    monitor = MemoryMonitor()
    monitor.start()
    try:
        t_start = time.time()
        adj, dead_mask, spider_nodes = load_graph(DATA_FILE, use_spider_check=True)
        n = adj.shape[0]
        spider_count = len(spider_nodes) if spider_nodes is not None else 0
        print(f"Detected {spider_count} spider nodes.")

        t_calc = time.time()
        pr = pagerank(adj, dead_mask)

        print(f"PageRank computed in {time.time() - t_calc:.2f}s")
        save_topk(pr)
    finally:
        monitor.stop()
        monitor.join()

    print(f"Peak memory: {monitor.peak / (1024 * 1024):.2f} MB")
    print(f"Total elapsed time: {time.time() - t_start:.2f}s")


if __name__ == '__main__':
    main()
