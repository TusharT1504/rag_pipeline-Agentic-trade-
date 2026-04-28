tushar@tushar:~/Documents/rag_pipeline$  source /home/tushar/Documents/rag_pipeline/.venv/bin/activate
(rag_pipeline) tushar@tushar:~/Documents/rag_pipeline$ python main.py query "What is the growth forecast for the ESDM sector?" --namespaces ESDM
2026-04-28T14:04:45 | INFO     | tools.answer_generation_tool | Sector analysis prompt loaded: Sector_analysis_prompt.yaml (3556 chars)
2026-04-28T14:04:45 | INFO     | __main__ | CLI query: What is the growth forecast for the ESDM sector?
2026-04-28T14:04:45 | INFO     | graph.rag_graph | RAG graph compiled successfully.
2026-04-28T14:04:45 | INFO     | graph.rag_graph | [router_node] Mode selected: retrieval
2026-04-28T14:04:45 | INFO     | observability.langsmith | [retrieval_node] START
2026-04-28T14:04:45 | INFO     | graph.nodes | [retrieval_node] Fetching all context from namespace(s)=['ESDM']…
2026-04-28T14:04:45 | INFO     | tools.retrieval_tool | Fetching namespace context directly | namespaces=['ESDM'] limit_per_namespace=all
2026-04-28T14:04:45 | INFO     | tools.vector_store_tool | Initialising Pinecone client…
2026-04-28T14:04:48 | INFO     | tools.vector_store_tool | Listed 16 vector ID(s) from namespace 'ESDM'
2026-04-28T14:04:49 | INFO     | tools.vector_store_tool | Fetched 16 chunk(s) from namespace 'ESDM'
2026-04-28T14:04:49 | INFO     | tools.vector_store_tool | Fetched 16 chunk(s) across 1 namespace(s)
2026-04-28T14:04:49 | INFO     | tools.retrieval_tool | Fetched 16 namespace context chunk(s).
2026-04-28T14:04:49 | INFO     | observability.langsmith | [retrieval_node] METADATA | {'ts': '2026-04-28T08:34:49.147650+00:00', 'node': 'retrieval_node', 'query': 'What is the growth forecast for the ESDM sector?', 'namespace': ['ESDM'], 'retrieved_docs_count': 16, 'retrieval_analysis': {'total_retrieved': 16, 'namespaces_hit': ['ESDM'], 'score_stats': {'min': 1.0, 'max': 1.0, 'avg': 1.0}, 'sources': ['ESDM.pdf']}}
2026-04-28T14:04:49 | INFO     | observability.langsmith | [retrieval_node] END | elapsed=4.094s
2026-04-28T14:04:49 | INFO     | observability.langsmith | [answer_generation_node] START
2026-04-28T14:04:49 | INFO     | graph.nodes | [answer_generation_node] Generating answer (chunks=16)…
2026-04-28T14:04:49 | INFO     | tools.answer_generation_tool | Initialising Groq client (model=llama-3.3-70b-versatile)…
2026-04-28T14:04:49 | INFO     | tools.answer_generation_tool | Calling Groq | model=llama-3.3-70b-versatile | sector=ESDM | chunks=16 | system_chars=3556 | user_chars=14075
2026-04-28T14:04:54 | INFO     | tools.answer_generation_tool | Report generated | attempt=1 | tokens=6910 | answer_chars=4478
2026-04-28T14:04:54 | INFO     | observability.langsmith | [answer_generation_node] METADATA | {'ts': '2026-04-28T08:34:54.298580+00:00', 'node': 'answer_generation_node', 'query': 'What is the growth forecast for the ESDM sector?', 'retrieved_docs_count': 16, 'answer_len': 4478, 'model': 'llama-3.3-70b-versatile'}
2026-04-28T14:04:54 | INFO     | observability.langsmith | [answer_generation_node] END | elapsed=5.150s

============================================================
ANSWER:
============================================================
## SECTOR OVERVIEW
The Indian ESDM industry is experiencing rapid growth, valued at approximately Rs. 2,09,000 crore (US$ 24.45 billion) in FY23, and is estimated to reach Rs. 9,09,000 crore (US$ 106.35 billion) by FY28, representing a compound annual growth rate (CAGR) of 34% (SOURCE FILE: ESDM.pdf, PAGE: 1). The sector is targeting a manufacturing output of Rs. 43,10,000 crore (US$ 500 billion) by FY30, requiring a fivefold increase in production, which is expected to create 12 million jobs by FY27 (SOURCE FILE: ESDM.pdf, PAGE: 1).

## MARKET STRUCTURE & KEY PLAYERS
The ESDM market is segmented into two main categories: Electronics System, which comprises 72.70% of the market, and Electronics Design, which makes up the remaining 27.30% (SOURCE FILE: ESDM.pdf, PAGE: 1). Major companies in the sector include Samsung, LG, Oppo, Vivo, Xiaomi, Panasonic, Midea, and TDK, among others (SOURCE FILE: ESDM.pdf, PAGE: 3). The market structure is characterized by the presence of several electronics manufacturing hubs, including NCR, Maharashtra, Karnataka, Tamil Nadu, Telangana, and Andhra Pradesh (SOURCE FILE: ESDM.pdf, PAGE: 3).

## GROWTH DRIVERS
The growth of the ESDM sector is driven by strong demand, supportive government policies, and increasing digitalization (SOURCE FILE: ESDM.pdf, PAGE: 1). The sector is also driven by the growth of the semiconductor market, which is expected to grow from Rs. 4,50,164 crore (US$ 52 billion) in FY24 to Rs. 8,95,134 crore (US$ 103.4 billion) by FY30 (SOURCE FILE: ESDM.pdf, PAGE: 2). Artificial Intelligence (AI) is also expected to add US$ 450-500 billion to India's GDP by 2025 (SOURCE FILE: ESDM.pdf, PAGE: 2).

## GOVERNMENT INITIATIVES & POLICY SUPPORT
The government has introduced several schemes to promote the sector, including the Production-Linked Incentive (PLI) scheme, which offers 4-6% incentives on incremental sales to boost domestic manufacturing of mobiles and components (SOURCE FILE: ESDM.pdf, PAGE: 3). The Scheme for Promotion of Manufacturing of Electronic Components and Semiconductors (SPECS) provides a 25% financial incentive on capital expenditure for identified electronic goods (SOURCE FILE: ESDM.pdf, PAGE: 4). The government has also announced the National Policy on Electronics (NPE) 2019, which envisions positioning India as a global hub for ESDM with a turnover target of US$ 400 billion by 2025 (SOURCE FILE: ESDM.pdf, PAGE: 4).

## INDUSTRY TRENDS & FUTURE OUTLOOK
The ESDM market is expected to grow at a CAGR of 16.1% to reach US$ 220 billion by FY25 (SOURCE FILE: ESDM.pdf, PAGE: 1). The electronics systems market is expected to grow at a CAGR of 14.8% to reach US$ 160 billion by FY25 (SOURCE FILE: ESDM.pdf, PAGE: 2). The electronics design market is expected to grow at a CAGR of 20.9% to reach US$ 60 billion by FY25 (SOURCE FILE: ESDM.pdf, PAGE: 2).

## RISK FACTORS / CHALLENGES
The sector faces challenges such as intense competition, high capital requirements, and the need for continuous innovation (SOURCE FILE: ESDM.pdf, PAGE: 3). The sector is also vulnerable to global economic trends and trade policies (SOURCE FILE: ESDM.pdf, PAGE: 4).

## INSIGHTFUL ANALYSIS
The growth of the ESDM sector is justified by the strong demand, supportive government policies, and increasing digitalization (SOURCE FILE: ESDM.pdf, PAGE: 1). The sector is expected to benefit from the growth of the semiconductor market and the increasing adoption of AI (SOURCE FILE: ESDM.pdf, PAGE: 2). The key players in the sector, including Samsung, LG, and Panasonic, are expected to drive growth and innovation (SOURCE FILE: ESDM.pdf, PAGE: 3). However, the sector faces challenges such as intense competition and high capital requirements (SOURCE FILE: ESDM.pdf, PAGE: 3).

## SUMMARY TABLE
| Factor | Insight | Supporting Evidence |
| --- | --- | --- |
| Market Size | Rs. 2,09,000 crore (US$ 24.45 billion) in FY23 | SOURCE FILE: ESDM.pdf, PAGE: 1 |
| Growth Rate | 34% CAGR to reach Rs. 9,09,000 crore (US$ 106.35 billion) by FY28 | SOURCE FILE: ESDM.pdf, PAGE: 1 |
| Key Players | Samsung, LG, Oppo, Vivo, Xiaomi, Panasonic, Midea, and TDK | SOURCE FILE: ESDM.pdf, PAGE: 3 |
| Government Initiatives | PLI scheme, SPECS, NPE 2019 | SOURCE FILE: ESDM.pdf, PAGE: 3-4 |
| Industry Trends | 16.1% CAGR to reach US$ 220 billion by FY25 | SOURCE FILE: ESDM.pdf, PAGE: 1 |
| Risk Factors | Intense competition, high capital requirements, global economic trends | SOURCE FILE: ESDM.pdf, PAGE: 3-4 |
(rag_pipeline) tushar@tushar:~/Documents/rag_pipeline$ 