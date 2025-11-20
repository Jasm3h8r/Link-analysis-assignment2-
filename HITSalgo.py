import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Streamlit Page Setup ---
st.set_page_config(page_title="PageRank & HITS Visualizer", layout="wide")
st.title("🌐 PageRank & HITS Network Visualizer")

# --- Sidebar Input ---
st.sidebar.header("Graph Setup")

# Option to upload dataset or create manually
upload_option = st.sidebar.radio("📂 Input Mode", ["Manual Entry", "Upload CSV"])

edge_list = []
nodes = []

if upload_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Edge List CSV", type=["csv"])
    st.sidebar.markdown("**Expected Columns:** `source,target`")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if {"source", "target"}.issubset(df.columns):
                edge_list = list(zip(df["source"], df["target"]))
                nodes = list(set(df["source"]) | set(df["target"]))
                st.success(f"✅ Loaded {len(nodes)} nodes and {len(edge_list)} edges from file.")
            else:
                st.error("CSV must contain 'source' and 'target' columns.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

else:
    num_nodes = st.sidebar.number_input("Number of Nodes", min_value=2, max_value=20, value=5)
    for i in range(num_nodes):
        node = st.sidebar.text_input(f"Name of Node {i+1}", value=f"Node{i+1}")
        nodes.append(node)

    edge_input = st.sidebar.text_area(
        "Enter Edges (source,target per line)",
        placeholder="Example:\nA,B\nB,C\nC,A"
    )

    for line in edge_input.strip().splitlines():
        try:
            src, tgt = line.strip().split(',')
            edge_list.append((src.strip(), tgt.strip()))
        except:
            pass

# --- Build Directed Graph ---
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edge_list)
N = len(G)

if len(G.edges()) == 0:
    st.warning("⚠️ Please add at least one edge to compute PageRank and HITS.")
    st.stop()

# --- PageRank Implementation ---
damping = 0.85  # (1 - 0.15)
convergence_threshold = 0.0001
pagerank = dict.fromkeys(G.nodes, 1.0 / N)

def pagerank_iteration(G, pagerank, damping=0.85):
    N = len(G)
    new_pr = dict.fromkeys(G.nodes(), (1 - damping) / N)
    for node in G:
        for n in G.predecessors(node):
            new_pr[node] += damping * pagerank[n] / len(G[n])
    return new_pr

def has_converged(pr1, pr2, threshold):
    return all(abs(pr1[n] - pr2[n]) < threshold for n in pr1)

while True:
    new_pr = pagerank_iteration(G, pagerank, damping)
    if has_converged(pagerank, new_pr, convergence_threshold):
        pagerank = new_pr
        break
    pagerank = new_pr

# --- HITS Implementation ---
def hits(G, max_iter=100, tol=1e-8):
    nodes = list(G.nodes())
    hub = dict.fromkeys(nodes, 1.0)
    auth = dict.fromkeys(nodes, 1.0)
    for _ in range(max_iter):
        last_auth, last_hub = auth.copy(), hub.copy()
        for n in nodes:
            auth[n] = sum(hub[j] for j in G.predecessors(n))
        norm = np.linalg.norm(list(auth.values()))
        for n in nodes:
            auth[n] /= norm
        for n in nodes:
            hub[n] = sum(auth[j] for j in G.successors(n))
        norm = np.linalg.norm(list(hub.values()))
        for n in nodes:
            hub[n] /= norm
        if (np.allclose(list(auth.values()), list(last_auth.values()), atol=tol) and
            np.allclose(list(hub.values()), list(last_hub.values()), atol=tol)):
            break
    return auth, hub

auth, hub = hits(G)

# --- Display Results ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 PageRank (Top 5)")
    st.table(sorted(pagerank.items(), key=lambda kv: kv[1], reverse=True)[:5])
with col2:
    st.subheader("🏆 HITS (Top 5)")
    st.markdown("**Authorities**")
    st.table(sorted(auth.items(), key=lambda kv: kv[1], reverse=True)[:5])
    st.markdown("**Hubs**")
    st.table(sorted(hub.items(), key=lambda kv: kv[1], reverse=True)[:5])

# --- Comparison Section ---
st.subheader("🔍 Comparison: PageRank vs HITS")

top5_pr = [n for n, _ in sorted(pagerank.items(), key=lambda kv: kv[1], reverse=True)[:5]]
top5_auth = [n for n, _ in sorted(auth.items(), key=lambda kv: kv[1], reverse=True)[:5]]
top5_hub = [n for n, _ in sorted(hub.items(), key=lambda kv: kv[1], reverse=True)[:5]]

overlap = set(top5_pr) & (set(top5_auth) | set(top5_hub))
if overlap:
    st.success(f"Nodes ranked high in both PageRank and HITS: {', '.join(overlap)}")
else:
    st.info("No strong overlap between top PageRank and HITS nodes.")

# --- Visualization ---
st.subheader("🎨 Graph Visualization")

pos = nx.spring_layout(G, seed=42)
sizes = [3000 * pagerank[n] for n in G.nodes()]
top_hubs = sorted(hub, key=hub.get, reverse=True)[:3]
top_auths = sorted(auth, key=auth.get, reverse=True)[:3]

plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=sizes, node_color="skyblue", edge_color="gray", font_size=10)
nx.draw_networkx_nodes(G, pos, nodelist=top_hubs, node_color="orange", label="Top Hubs")
nx.draw_networkx_nodes(G, pos, nodelist=top_auths, node_color="green", label="Top Authorities")
plt.legend()
plt.title("PageRank & HITS Visualization")
st.pyplot(plt)

st.markdown("### 🧠 Interpretation")
st.write("""
In this network, PageRank identifies the most trusted or frequently referenced nodes, 
while HITS distinguishes between authoritative sources and influential connectors (hubs). 
Nodes common to both represent entities that are both highly cited and highly connected.
""")

