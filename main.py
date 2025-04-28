import argparse
import numpy as np
import scipy.sparse as sp
import time
import psutil
import os
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import connected_components
import threading

TOL = 1e-6


def load_graph(filename, use_spider_check=False, spider_threshold=1):
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

    spider_nodes = None
    if use_spider_check:
        # 出度小于等于阈值视为蜘蛛节点
        spider_nodes = np.where(out_degree <= spider_threshold)[0]
        print(f"Found {len(spider_nodes)} potential spider nodes (out_degree <= {spider_threshold})")

    return adj, (out_degree == 0), spider_nodes


def pagerank_spider_trap(adj, alpha=0.85, tol=TOL, max_iter=50000):
    """带蜘蛛陷阱检测和处理的PageRank"""
    n = adj.shape[0]
    pr = np.full(n, 1.0 / n, dtype=np.float32)
    teleport = (1 - alpha) / n

    # 计算出度及死节点
    out_degree = adj.sum(axis=0).A1.astype(np.float32)
    dangling = out_degree == 0

    # 强连通分量检测蜘蛛陷阱
    n_comp, labels = connected_components(adj, directed=True, connection="strong")
    trap_mask = np.zeros(n, dtype=bool)
    for comp in range(n_comp):
        nodes = np.where(labels == comp)[0]
        sub = adj[nodes, :][:, nodes]
        # 如果子图没有出边，则整个连通分量是陷阱
        if sub.sum() == adj[nodes, :].sum():
            trap_mask[nodes] = True
    print(f"Detected {trap_mask.sum()} spider-trap nodes via strong components")

    for i in range(max_iter):
        old = pr.copy()
        # 处理死节点与陷阱节点
        dangling_sum = alpha * old[dangling].sum()
        trap_sum = alpha * old[trap_mask].sum()

        # 基本迭代
        pr = alpha * adj.dot(old)
        # 添加 teleport
        pr += teleport
        # 添加额外处理：死节点及蜘蛛陷阱贡献各自均匀分布
        pr += dangling_sum / n + trap_sum / n

        # 归一化
        pr /= pr.sum()

        # 收敛检测
        if np.linalg.norm(pr - old, 1) < tol:
            print(f"[SpiderTrap] Converged at {i+1} iterations, delta={np.linalg.norm(pr-old,1):.2e}")
            break
    return pr


def pagerank(adj, dead_mask, alpha=0.85, tol=TOL, max_iter=50000):
    """标准Pagerank，稀疏矩阵直接乘"""
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
        np.multiply(pr_next, alpha_f32, out=pr_next)
        pr_next += alpha_f32 * dead_sum / n + teleport_f32
        if np.abs(pr_next - pr).sum() < tol:
            print(f"Converged at {it+1} iterations, delta={np.abs(pr_next-pr).sum():.2e}")
            break
        pr, pr_next = pr_next, pr
    return pr


def block_pagerank(adj, dead_mask, alpha=0.85, tol=TOL, max_iter=50000, block_size=None):
    """Block Matrix版PageRank（分块大小就是数据集大小）"""
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
    parser.add_argument("--use_spider_check", type=int, default=0, help="Use spider check")
    parser.add_argument("--use_block", type=int, default=0, help="Use block matrix")
    args = parser.parse_args()

    global TOL
    TOL = args.tol

    monitor = MemoryMonitor()
    monitor.start()
    try:
        t_start = time.time()
        # 加载图并检测蜘蛛节点
        adj, dead_mask, spider_nodes = load_graph(args.input, use_spider_check=args.use_spider_check)
        n = adj.shape[0]
        spider_count = len(spider_nodes) if spider_nodes is not None else 0
        print(f"Detected {spider_count} spider nodes.")

        # 动态计算阈值：取总节点数的5%和1000中的较大者
        threshold = max(1000, int(n * 0.05))
        # 自动选择是否启用蜘蛛陷阱处理
        # use_spider_trap = spider_count > threshold
        use_spider_trap = args.use_spider_check
        # 是否启用Block Matrix（可根据需要添加自动逻辑）
        use_block = args.use_block

        print(
            f"Total nodes: {n}, Threshold: {threshold}, Use Spider Trap Handling: {use_spider_trap == 1}, Use Block Matrix: {use_block == 1}"
        )

        # 选择算法
        t_calc = time.time()
        if use_spider_trap:
            pr = pagerank_spider_trap(adj)
        elif use_block:
            pr = block_pagerank(adj, dead_mask)
        else:
            pr = pagerank(adj, dead_mask)

        print(f"PageRank computed in {time.time()-t_calc:.2f}s")
        save_topk(pr, filename=args.output)
    finally:
        monitor.stop()
        monitor.join()
    print(f"峰值内存: {monitor.peak/(1024 * 1024):.2f}MB")
    print(f"总耗时: {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    main()
