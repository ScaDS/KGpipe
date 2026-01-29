

# Relation Linker

```
SBERT embeddings of relation phrases
   ↓
Cluster phrases
   ↓
LLM or rule-based mapping of clusters → KG predicates
```

# Entity Linker

For every node in your KG, you create a textual profile:

```
Node ID: E_1423
Label: "Tumor necrosis factor"
Aliases: ["TNF", "TNF-alpha"]
Description: "Cytokine involved in systemic inflammation..."
Types: ["protein", "cytokine"]
```

Now entity linking becomes text similarity search.


The modern approach (used in papers and industry)
Step 1 — Embed all KG entities

Use SBERT / E5 / BGE to embed:

[label + aliases + description + types]


Store in a vector index (FAISS, Milvus, etc.)

Step 2 — For each OpenIE argument span

Example:

“TNF-alpha”

Embed the mention (plus sentence context if you want).

Step 3 — Nearest neighbor search in your KG

Retrieve top-k candidate nodes from your KG.

This replaces “Wikipedia candidate generation”.

Step 4 — Cross-encoder or LLM re-ranking (optional but powerful)

Score:

(mention, sentence context, candidate entity description)

Pick the best.

This is exactly what BLINK/REL do — just with Wikipedia.
You do the same with your KG.

SBERT + FAISS + cross-encoder
## Issue
new profile is not that similar?


# Issues
inverse predicates... require typ detection before linking