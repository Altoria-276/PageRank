#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

// PageRank algorithm implementation
vector<double> pagerank(const vector<vector<int>>& adj,
                        double damping = 0.85,
                        double eps = 1e-6,
                        int maxIter = 100) {
    int N = adj.size();
    vector<double> pr(N, 1.0 / N), pr_new(N, 0.0);
    vector<int> out_deg(N);
    double base = (1.0 - damping) / N;

    // calculate out-degree for each node
    for (int i = 0; i < N; ++i) {
        out_deg[i] = adj[i].size();
    }

    for (int iter = 0; iter < maxIter; ++iter) {
        // initialize new PR values
        for (int i = 0; i < N; ++i)
            pr_new[i] = base;

        double dangling_sum = 0.0;
        // handle dangling nodes and distribute rank
        for (int i = 0; i < N; ++i) {
            if (out_deg[i] == 0) {
                dangling_sum += pr[i];
            } else {
                double share = pr[i] / out_deg[i];
                for (int v : adj[i]) {
                    pr_new[v] += damping * share;
                }
            }
        }
        // distribute dangling node contribution
        double dangling_contrib = damping * dangling_sum / N;
        for (int i = 0; i < N; ++i) {
            pr_new[i] += dangling_contrib;
        }

        // check convergence
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
    // read edge list from Data.txt, each line: u v
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

    // build graph, assuming nodes are 0..maxNode
    int N = maxNode + 1;
    vector<vector<int>> graph(N);
    for (auto& e : edges) {
        graph[e.first].push_back(e.second);
    }

    // parameters
    double damping = 0.85;
    double eps = 1e-6;
    int maxIter = 100000;

    // compute PageRank
    vector<double> pr = pagerank(graph, damping, eps, maxIter);

    // write results to Res.txt
    ofstream outfile("Res.txt");
    if (!outfile) {
        cerr << "Failed to open file Res.txt" << endl;
        return 1;
    }
    outfile << "Node\tPageRank" << endl;
    for (int i = 0; i < pr.size(); ++i) {
        outfile << i << "\t" << pr[i] << endl;
    }
    outfile.close();

    return 0;
}
