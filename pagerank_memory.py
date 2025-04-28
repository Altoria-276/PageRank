import numpy as np
import scipy.sparse as sp
import time
import psutil
import os
from scipy.sparse import csc_matrix

from pagerank import pagerank_advanced


# === 使用最小内存加载整图为 CSC 矩阵 ===
def load_graph(filename):
    """
    优化：单遍扫描构建单个 CSC，避免分块带来的冗余存储。
    """
    rows, cols = [], []
    max_node = 0
    with open(filename, "r") as f:
        for line in f:
            src, dst = map(int, line.split())
            rows.append(dst)
            cols.append(src)
            if src > max_node:
                max_node = src
            if dst > max_node:
                max_node = dst
    n = max_node + 1
    data = np.ones(len(rows), dtype=np.float32)
    adj = csc_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return adj


# === 原地 Gauss–Seidel 迭代，收敛与内存节省 ===
def pagerank(adj, alpha=0.85, tol=1e-6, max_iter=10000):
    """
    1. 结合死节点和 teleport：预计算标量
    2. Gauss–Seidel 就地更新：无需双向量和额外缓冲
    3. 直接遍历非零列加速：利用 CSC 索引快速获取邻居
    """
    n = adj.shape[0]
    out_deg = np.array(adj.sum(axis=0)).astype(np.float32).ravel()
    inv_out = np.zeros(n, dtype=np.float32)
    mask = out_deg > 0
    inv_out[mask] = 1.0 / out_deg[mask]
    dead_idx = np.where(~mask)[0]

    pr = np.full(n, 1.0 / n, dtype=np.float32)
    teleport_const = (1 - alpha) / n
    col_ptrs = adj.indptr
    row_idx = adj.indices
    data = adj.data

    for it in range(max_iter):
        old_pr = pr.copy()
        dangling = old_pr[dead_idx].sum()
        base = alpha * dangling / n + teleport_const
        for i in range(n):
            start, end = col_ptrs[i], col_ptrs[i + 1]
            s = 0.0
            for idx in range(start, end):
                j = row_idx[idx]
                s += data[idx] * inv_out[j] * pr[j]
            pr[i] = alpha * s + base
        if np.abs(pr - old_pr).sum() < tol:
            print(f"在第 {it+1} 次迭代收敛")
            break
    return pr


# === 高效 Top-K 节点选择 ===
def save_topk(pr, topk=100, filename="Res.txt"):
    topk_idxs = np.argpartition(-pr, topk)[:topk]
    topk_sorted = topk_idxs[np.argsort(-pr[topk_idxs])]
    with open(filename, "w") as f:
        for idx in topk_sorted:
            f.write(f"{idx} {pr[idx]:.8f}\n")


# === 主程序：监控峰值内存 & 运行时间 ===
import threading

# 全局变量用于存储峰值内存
peak_rss = 0
monitoring = True


def memory_monitor(interval=0.01):
    """
    后台线程定期采样当前进程 RSS，记录峰值。
    """
    global peak_rss, monitoring
    proc = psutil.Process(os.getpid())
    while monitoring:
        try:
            rss = proc.memory_info().rss / (1024 * 1024)
            if rss > peak_rss:
                peak_rss = rss
        except psutil.Error:
            break
        time.sleep(interval)


def main():
    global monitoring, peak_rss
    # 启动内存监控线程
    monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
    monitor_thread.start()

    start_time = time.time()
    adj = load_graph("Data.txt")
    pr = pagerank(adj, alpha=0.85)
    # pr = pagerank_advanced(adj, alpha=0.85)
    save_topk(pr)
    elapsed = time.time() - start_time

    # 停止监控
    monitoring = False
    monitor_thread.join()

    print(f"完成于 {elapsed:.2f}s，峰值内存: {peak_rss:.2f} MB")


if __name__ == "__main__":
    main()
