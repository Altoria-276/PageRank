import numpy as np
import scipy.sparse as sp
import time
import psutil
import os
import threading
from scipy.sparse import csc_matrix

# 内存监控全局变量
peak_rss = 0
monitoring = True


def memory_monitor(interval=0.01):
    """内存监控线程"""
    global peak_rss, monitoring
    proc = psutil.Process(os.getpid())
    while monitoring:
        try:
            current_rss = proc.memory_info().rss / (1024 * 1024)
            if current_rss > peak_rss:
                peak_rss = current_rss
        except psutil.Error:
            break
        time.sleep(interval)


def load_graph(filename):
    """单次IO优化加载图并可选检测蜘蛛节点"""
    data = np.loadtxt(filename, dtype=np.int32)
    src = data[:, 0]
    dst = data[:, 1]
    max_node = max(src.max(), dst.max())
    n = max_node + 1

    out_degree = np.bincount(src, minlength=n).astype(np.float32)
    np.maximum(out_degree, 1.0, out=out_degree)  # 避免除零

    inv_degree = 1.0 / out_degree[src]
    adj = csc_matrix((inv_degree, (dst, src)), shape=(n, n), dtype=np.float32)
    adj.sum_duplicates()

    return adj


def pagerank(adj, alpha=0.85, tol=1e-6, max_iter=1000000):
    """原始矩阵运算实现"""
    n = adj.shape[0]
    out_degree = np.array(adj.sum(axis=0)).flatten().astype(float)
    dead_ends = out_degree == 0

    # 保持原始归一化方式
    inv_out = np.reciprocal(out_degree, where=out_degree != 0)
    D_inv = sp.diags(inv_out)
    M = adj @ D_inv

    pr = np.ones(n) / n  # 保持float64精度
    teleport = np.ones(n) / n

    for _ in range(max_iter):
        pr_new = alpha * (M @ pr + np.sum(pr[dead_ends]) / n) + (1 - alpha) * teleport
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break
        pr = pr_new
    return pr


def save_topk(pr, topk=100, filename="Res.txt"):
    """原始排序方式"""
    top_indices = np.argsort(-pr)[:topk]  # 完全排序
    with open(filename, "w") as f:
        for idx in top_indices:
            f.write(f"{idx} {pr[idx]:.8f}\n")


def main():
    global monitoring, peak_rss

    # 启动内存监控线程
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()

    start_time = time.time()

    # 保持原始处理流程
    adj = load_graph("Data.txt")
    pr = pagerank(adj, alpha=0.85)
    save_topk(pr)

    elapsed = time.time() - start_time

    # 停止监控
    monitoring = False
    monitor_thread.join()

    print(f"总耗时: {elapsed:.2f}s")
    print(f"峰值内存: {peak_rss:.2f} MB")


if __name__ == "__main__":
    main()
