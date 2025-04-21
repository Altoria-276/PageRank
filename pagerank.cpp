#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// SparseMatrix class using CSR format
class SparseMatrix {
public:
    int rows, cols;
    vector<pair<int, int>> entries;
    vector<int> row_ptr;
    vector<int> col_idx;

    SparseMatrix(int r, int c) : rows(r), cols(c) {}

    // Add a non-zero entry at (r, c)
    void addEntry(int r, int c) {
        entries.emplace_back(r, c);
    }

    // Build CSR structure
    void finalize() {
        row_ptr.assign(rows + 1, 0);
        // Count entries per row
        for (auto &e : entries) row_ptr[e.first + 1]++;
        // Cumulative sum to get row_ptr
        for (int i = 0; i < rows; ++i) row_ptr[i + 1] += row_ptr[i];

        col_idx.resize(entries.size());
        vector<int> counter = row_ptr;
        // Fill column indices
        for (auto &e : entries) {
            int r = e.first;
            col_idx[counter[r]++] = e.second;
        }
    }

    // Number of non-zeros in row r
    int outDegree(int r) const {
        return row_ptr[r + 1] - row_ptr[r];
    }

    // Get column indices (neighbors) for row r
    vector<int> neighbors(int r) const {
        vector<int> res;
        for (int i = row_ptr[r]; i < row_ptr[r + 1]; ++i) {
            res.push_back(col_idx[i]);
        }
        return res;
    }
};

// PageRank algorithm implementation using SparseMatrix
vector<double> pagerank(const SparseMatrix &mat,
                        double damping = 0.85,
                        double eps = 1e-6,
                        int maxIter = 100) {
    int N = mat.rows;
    vector<double> pr(N, 1.0 / N), pr_new(N, 0.0);
    double base = (1.0 - damping) / N;

    for (int iter = 0; iter < maxIter; ++iter) {
        // Initialize new PageRank values
        for (int i = 0; i < N; ++i) pr_new[i] = base;

        double dangling_sum = 0.0;
        // Distribute rank
        for (int i = 0; i < N; ++i) {
            int out_deg = mat.outDegree(i);
            if (out_deg == 0) {
                dangling_sum += pr[i];
            } else {
                double share = pr[i] / out_deg;
                for (int v : mat.neighbors(i)) {
                    pr_new[v] += damping * share;
                }
            }
        }
        // Distribute dangling contributions evenly
        double dangling_contrib = damping * dangling_sum / N;
        for (int i = 0; i < N; ++i) pr_new[i] += dangling_contrib;

        // Check convergence
        double diff = 0.0;
        for (int i = 0; i < N; ++i) {
            diff = max(diff, fabs(pr_new[i] - pr[i]));
            pr[i] = pr_new[i];
        }
        if (diff < eps) break;
    }
    return pr;
}

int main() {
    // Read edge list from Data.txt (each line: u v)
    ifstream infile("Data.txt");
    if (!infile) {
        cerr << "Failed to open file Data.txt" << endl;
        return 1;
    }

    vector<pair<int, int>> edges;
    int u, v;
    int maxNode = -1;
    while (infile >> u >> v) {
        edges.emplace_back(u, v);
        maxNode = max(maxNode, max(u, v));
    }
    infile.close();
    int N = maxNode + 1;

    // Build sparse matrix
    SparseMatrix mat(N, N);
    for (auto &e : edges) mat.addEntry(e.first, e.second);
    mat.finalize();

    // PageRank parameters
    double damping = 0.85;
    double eps = 1e-6;
    int maxIter = 1000000;

    // Compute PageRank
    vector<double> pr = pagerank(mat, damping, eps, maxIter);

    // Pair ranks with node indices
    vector<pair<double, int>> pr_with_idx;
    pr_with_idx.reserve(N);
    for (int i = 0; i < N; ++i) {
        pr_with_idx.emplace_back(pr[i], i);
    }
    // Sort descending by PageRank value
    sort(pr_with_idx.begin(), pr_with_idx.end(),
         [](const pair<double, int> &a, const pair<double, int> &b) {
             return a.first > b.first;
         });

    // Write top 100 results to Res.txt
    ofstream outfile("Res.txt");
    if (!outfile) {
        cerr << "Failed to open file Res.txt" << endl;
        return 1;
    }

    int topK = min(100, N);
    for (int i = 0; i < topK; ++i) {
        int node = pr_with_idx[i].second;
        double score = pr_with_idx[i].first;
        outfile << node << " " << score << endl;
    }
    outfile.close();

    return 0;
}
