# Knowledge base (not committed)

Place the DeepSC paper text at:

```text
rag_agent/data/xie2021.txt
```

Suggested source (arXiv preprint of Xie et al., IEEE TSP 2021):

- https://arxiv.org/abs/2006.10685

Extract or paste the sections you want the RAG agent to retrieve (abstract, introduction, system model, experiments, etc.), save as UTF-8 plain text, then run:

```bash
python -m rag_agent.ingest
```

`xie2021.txt` is gitignored so each user supplies their own copy.
