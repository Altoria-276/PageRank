import numpy as np
import scipy.sparse as sp
import time

def load_graph(filename):
    rows = []
    cols = []
    with open(filename, 'r') as f:
        for line in f:
            src, dst = map(int, line.strip().split())
            rows.append(dst)
            cols.append(src)
    data = np.ones(len(rows))
    n_nodes = max(max(rows), max(cols)) + 1
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsc()
    return adj

def pagerank(adj, alpha=0.85, tol=1e-6, max_iter=1000000):
    n = adj.shape[0]
    out_degree = np.array(adj.sum(axis=0)).flatten()
    dead_ends = (out_degree == 0)

    # Normalize columns
    inv_out = np.reciprocal(out_degree, where=out_degree != 0)
    D_inv = sp.diags(inv_out)
    M = adj @ D_inv

    pr = np.ones(n) / n
    teleport = np.ones(n) / n

    for _ in range(max_iter):
        pr_new = alpha * (M @ pr + np.sum(pr[dead_ends]) / n) + (1 - alpha) * teleport
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break
        pr = pr_new
    return pr

def save_topk(pr, topk=100, filename='Res.txt'):
    top_indices = np.argsort(-pr)[:topk]
    with open(filename, 'w') as f:
        for idx in top_indices:
            f.write(f"{idx} {pr[idx]:.8f}\n")

def main():
    start_time = time.time()
    adj = load_graph("Data.txt")
    pr = pagerank(adj, alpha=0.85)
    save_topk(pr)
    print(f"Finished in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
