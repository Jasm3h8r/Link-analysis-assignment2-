# Link-analysis-assignment2

## Overview
This project implements and compares two fundamental link-analysis algorithms: **PageRank** and **HITS (Hyperlink-Induced Topic Search)**. Both algorithms measure the importance of nodes in a directed graph, but they use different approaches. This project allows you to build a graph, visualize it, run the algorithms, and observe how the scores differ.

---

## PageRank Algorithm

### Purpose
PageRank assigns a single importance score to each node based on the structure of incoming links.

### How It Works
- All nodes start with equal rank.
- At each iteration:
  - A node distributes its rank equally among all nodes it links to.
  - A fraction of the rank (teleportation) is distributed uniformly to all nodes.
- The process repeats until the scores converge.

### Damping Factor
If the damping factor is **n**, the teleportation probability becomes **1 − n**.

### Update Formula
```text
PR(i) = n * Σ(PR(j) / outdegree(j)) + (1 - n) / N
