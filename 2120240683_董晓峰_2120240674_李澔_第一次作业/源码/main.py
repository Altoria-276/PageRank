import argparse
import numpy as np
import scipy.sparse as sp


def load_graph(filename) -> sp.csc_array:
    rows, cols = [], []
    with open(filename, "r") as f:
        for line in f:
            u, v = map(int, line.strip().split())
            rows.append(v)
            cols.append(u)
    data = np.ones(len(rows))
    n = max(max(rows), max(cols)) + 1
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsc()
    return adj


def pagerank_advanced(adj: sp.csc_array, alpha=0.85, beta=0.0, tol=1e-6, max_iter=1000):
    n = adj.shape[0]
    out_degree = np.array(adj.sum(axis=0)).flatten()
    dangling = out_degree == 0

    # 构建稀疏列归一化转移矩阵
    inv_out = np.reciprocal(out_degree, where=out_degree != 0)
    D_inv = sp.diags(inv_out)
    M = adj @ D_inv  # 即列归一化

    r = np.ones(n) / n
    teleport = np.ones(n) / n  # 可替换为其他个性化向量

    for _ in range(max_iter):
        dead_end_mass = np.sum(r[dangling]) / n
        r_new = alpha * (M @ r + dead_end_mass) + beta * r + (1 - alpha - beta) * teleport

        delta = np.linalg.norm(r_new - r, 1)
        if delta < tol:
            break
        r = r_new

    return r


def save_topk(ranks, topk=100, filename="Res.txt"):
    top_nodes = np.argsort(-ranks)[:topk]
    with open(filename, "w") as f:
        for i in top_nodes:
            f.write(f"{i} {ranks[i]:.8f}\n")


def main():
    parser = argparse.ArgumentParser(description="Advanced PageRank algorithm implementation")
    parser.add_argument("--input", default="Data.txt", help="Input file path")
    parser.add_argument("--output", default="Res.txt", help="Output file path")
    parser.add_argument("--alpha", type=float, default=0.85, help="Damping factor")
    parser.add_argument("--beta", type=float, default=0.05, help="Current rank retention ratio")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence threshold")
    parser.add_argument("--max_iter", type=int, default=1000, help="Maximum iterations")
    args = parser.parse_args()

    adj = load_graph(args.input)
    ranks = pagerank_advanced(adj, alpha=args.alpha, beta=args.beta, tol=args.tol, max_iter=args.max_iter)
    save_topk(ranks, filename=args.output)


if __name__ == "__main__":
    main()
