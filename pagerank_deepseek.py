import numpy as np
import time


def load_graph(filename):
    edges = []
    nodes = set()
    with open(filename, "r") as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            nodes.add(src)
            nodes.add(dst)
            edges.append((src, dst))

    # 映射节点到连续索引
    node_list = sorted(nodes)
    n = len(node_list)
    node_id_to_idx = {node: idx for idx, node in enumerate(node_list)}

    # 转换边为索引并统计邻接信息
    out_degree = np.zeros(n, dtype=np.int32)
    adj_list = [[] for _ in range(n)]
    for src, dst in edges:
        src_idx = node_id_to_idx[src]
        dst_idx = node_id_to_idx[dst]
        out_degree[src_idx] += 1
        adj_list[src_idx].append(dst_idx)

    # 分块处理源节点（块大小根据内存调整）
    block_size = 1000
    source_nodes = np.where(out_degree > 0)[0].tolist()
    blocks = [source_nodes[i : i + block_size] for i in range(0, len(source_nodes), block_size)]

    # 收集dead ends
    dead_ends = np.where(out_degree == 0)[0].tolist()

    return {"n": n, "blocks": blocks, "adj_list": adj_list, "out_degree": out_degree, "dead_ends": dead_ends, "node_list": node_list}


def pagerank(n, blocks, adj_list, out_degree, dead_ends, alpha=0.85, tol=1e-6, max_iter=1000):
    pr = np.ones(n, dtype=np.float32) / n
    teleport_value = (1 - alpha) / n

    for _ in range(max_iter):
        pr_new = np.full(n, teleport_value, dtype=np.float32)
        # 处理每个块中的源节点
        for block in blocks:
            for src in block:
                contrib = alpha * pr[src] / out_degree[src]
                for dst in adj_list[src]:
                    pr_new[dst] += contrib

        # 处理dead ends的贡献
        dead_contrib = alpha * np.sum(pr[dead_ends])
        pr_new += dead_contrib / n

        # 检查收敛
        delta = np.abs(pr_new - pr).sum()
        if delta < tol:
            break
        pr = pr_new.copy()

    return pr


def save_topk(pr, node_list, filename="Res.txt", topk=100):
    top_indices = np.argsort(-pr)[:topk]
    with open(filename, "w") as f:
        for idx in top_indices:
            f.write(f"{node_list[idx]} {pr[idx]:.8f}\n")


def main():
    data = load_graph("Data.txt")
    pr = pagerank(data["n"], data["blocks"], data["adj_list"], data["out_degree"], data["dead_ends"], alpha=0.85)
    save_topk(pr, data["node_list"])


if __name__ == "__main__":
    main()
