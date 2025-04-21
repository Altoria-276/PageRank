### 数据集

数据文件为 Data.txt，包含 10000 个结点（0 - 9999，其中部分结点孤立，如 9991），含 150000 条边。

### 关键代码

```python
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
```

M 为转移矩阵，迭代方程为
$$
r_{new}=\alpha\cdot(M\times r+D)+\beta\cdot r+(1-\alpha-\beta)\cdot T
$$
其中 D 矩阵用于解决 Dead Ends 问题，即 Dead Ends 结点随机跳转至任意结点。

T 矩阵用于解决 Spider Traps 问题，用于一定概率的全局随机跳转避免蜘蛛陷阱。

### 内存优化

数据矩阵通过 CSC 存储，使用 scipy.sparse.csc_array

### 结果分析

部分输出结果如下，TOP 10 的结点。分析 Data.txt 可知 286 结点入度为 30，出度为 20。

```
286 0.000195649
3473 0.000193257
4951 0.000190095
3890 0.000188691
7365 0.000186658
6359 0.00018527
4352 0.000179715
7032 0.000179217
7541 0.000178745
3699 0.000178184
```

