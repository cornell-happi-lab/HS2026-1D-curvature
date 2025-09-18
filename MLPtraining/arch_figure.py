# save as make_arch_diagram.py
from graphviz import Digraph

g = Digraph("MLP", format="png")
g.attr(rankdir="LR", splines="spline", nodesep="0.4", ranksep="0.6")
box = {"shape":"box", "style":"rounded,filled", "color":"#666666"}

g.node("in",  "Input\n(16 nodes + 8 feats)", fillcolor="#f3f3f3", **box)
g.node("stem","Stem\nLinear(24→128)\nLayerNorm + SiLU + Dropout", fillcolor="#cfe8ff", **box)
g.node("rb1", "ResBlock ×1\nLinear(128)×2\nSiLU + Dropout + skip", fillcolor="#ffdfb8", **box)
g.node("rb2", "ResBlock ×1", fillcolor="#ffdfb8", **box)
g.node("rb3", "ResBlock ×1", fillcolor="#ffdfb8", **box)
g.node("head","Head\nLinear(128→64) → SiLU+Dropout → Linear(64→1)", fillcolor="#cfead6", **box)
g.node("out","Output\n$\hat{\\kappa}$ (1/m)", fillcolor="#f3f3f3", **box)

for a,b in [("in","stem"),("stem","rb1"),("rb1","rb2"),("rb2","rb3"),("rb3","head"),("head","out")]:
    g.edge(a,b, penwidth="2")

g.render("mlp_architecture", cleanup=True)  # creates mlp_architecture.png
print("Saved mlp_architecture.png")
