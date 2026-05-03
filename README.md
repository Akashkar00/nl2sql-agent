# NL2SQL-Agent: Self-Correcting Text-to-SQL with Fine-Tuned Qwen2.5-Coder

> A natural-language-to-SQL system that doesn't just generate queries — it executes them, catches its own errors, and retries. Fine-tuned on Spider + BIRD with QLoRA, deployed as a FastAPI service with a Streamlit demo.

---

## Why This Project (The Honest Pitch)

99% of student text-to-SQL projects do this: fine-tune CodeLlama on Spider, report execution accuracy, deploy a FastAPI endpoint, done. Recruiters have seen 200 of these. They blend in.

This project differs in three ways that interviewers actually care about:

1. **BIRD benchmark, not just Spider.** BIRD has dirty schemas, ambiguous columns, and external knowledge requirements — closer to real enterprise data. Most candidates have never run BIRD.
2. **Self-correcting execution loop.** When generated SQL fails (syntax error, empty result, type mismatch), the model gets the error and retries. This is how production systems like Vanna and Snowflake Cortex actually work.
3. **Schema-aware retrieval for large DBs.** Models can't fit a 200-table schema into context. We use embedding-based retrieval to pick relevant tables first.

If you build only the vanilla version, you have a portfolio project. If you build all three, you have something to talk about for 30 minutes in an interview.

---

## Architecture

```
User Question (NL)
       ↓
[Schema Retriever]  ← retrieves top-k relevant tables via embeddings
       ↓
[Fine-tuned Qwen2.5-Coder-7B + QLoRA adapter]
       ↓
   SQL Query
       ↓
[Executor on SQLite/Postgres]
       ↓
   ┌───┴───┐
Success?  Error/Empty?
   ↓        ↓
Return   [Self-Correction Loop]  ← max 2 retries with error feedback
         ↓
       Return final result or graceful failure
```

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Base Model | Qwen2.5-Coder-7B-Instruct | Beats CodeLlama-7B on SQL by ~10pts; current SOTA in the 7B class |
| Fine-tuning | QLoRA via Unsloth | 2× faster + half the memory of vanilla PEFT; critical on free Colab |
| Quantization | bitsandbytes 4-bit | Fits 7B model in ~6GB VRAM for inference |
| Embeddings (retrieval) | BAAI/bge-small-en-v1.5 | Small, fast, free, strong on schema text |
| Vector store | FAISS (local) | No infra, runs on Colab |
| Execution engine | SQLite (Spider/BIRD use it) + SQLAlchemy | Spider/BIRD ship as SQLite DBs |
| Backend | FastAPI + Uvicorn | Industry standard for ML services |
| Demo UI | Streamlit | Fastest path to a shareable demo |
| Experiment tracking | Weights & Biases (free tier) | Recruiters can see your training runs publicly |
| Deployment | HuggingFace Spaces (Streamlit) + ngrok for FastAPI demo | Free, public-facing |

**Compute reality:** Free Colab (T4 16GB) works but disconnects randomly during long training runs. Spend $10 on **Colab Pro** for the fine-tuning month — the time saved fighting disconnects is worth it. Alternative: **Kaggle** (T4×2, 30hr/week, more stable than free Colab).

---

## Project Structure

```
nl2sql-agent/
├── data/
│   ├── spider/                  # Spider dataset (download, gitignored)
│   ├── bird/                    # BIRD dataset (download, gitignored)
│   └── processed/               # Instruction-formatted train/eval JSONL
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_zero_shot.ipynb
│   ├── 03_finetune_qlora.ipynb     # Main training notebook (Colab)
│   └── 04_eval_and_error_analysis.ipynb
├── src/
│   ├── data/
│   │   ├── prepare_dataset.py
│   │   └── schema_formatter.py
│   ├── model/
│   │   ├── inference.py
│   │   └── load_adapter.py
│   ├── retrieval/
│   │   └── schema_retriever.py
│   ├── agent/
│   │   ├── executor.py
│   │   └── self_correction.py     # The retry loop
│   ├── eval/
│   │   ├── execution_accuracy.py
│   │   └── run_benchmark.py
│   └── api/
│       └── main.py                # FastAPI app
├── app/
│   └── streamlit_demo.py
├── configs/
│   └── training_config.yaml
├── adapters/                       # Saved LoRA adapters (gitignored, push to HF Hub)
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## Prerequisites

- Python 3.10+
- Comfortable with PyTorch + HuggingFace (you've fine-tuned before — good)
- A HuggingFace account (for hosting your adapter)
- A Weights & Biases account (free)
- Colab Pro recommended, or Kaggle account
- Git and basic Docker

---

## Phase-by-Phase Build Plan (6–8 weeks realistic)

### Phase 1 — Setup & Baseline (Week 1)

**Goal:** Have working environment + a number to beat before you fine-tune anything.

1. Clone Spider: https://yale-lily.github.io/spider
2. Download BIRD: https://bird-bench.github.io/
3. Set up Colab notebook, mount Drive, install deps:
   ```bash
   pip install -q unsloth transformers==4.44.0 trl==0.10.0 peft==0.12.0 \
                  bitsandbytes==0.43.3 accelerate==0.33.0 datasets==2.20.0 \
                  faiss-cpu sentence-transformers sqlalchemy wandb
   ```
4. Run **zero-shot baseline**: load Qwen2.5-Coder-7B-Instruct, run on Spider dev set (1034 examples), compute execution accuracy. Expect ~55–65% zero-shot.
5. **Write down the number.** This is your floor. If fine-tuning doesn't beat this by ≥8 points, something is wrong.

**Deliverable:** `notebooks/02_baseline_zero_shot.ipynb` with logged baseline metrics.

### Phase 2 — Data Preparation (Week 2)

**Goal:** Clean instruction-formatted training set with proper schema serialization.

1. **Format each example as instruction-tuning input:**
   ```
   ### Schema:
   CREATE TABLE customers (id INT, name VARCHAR, country VARCHAR);
   CREATE TABLE orders (id INT, customer_id INT, total DECIMAL, order_date DATE);
   -- 3 sample rows from customers: ...
   
   ### Question: How many customers are from Germany?
   
   ### SQL:
   SELECT COUNT(*) FROM customers WHERE country = 'Germany';
   ```
2. **Schema serialization matters more than people think.** Include: CREATE TABLE statements + 3 sample rows + foreign key hints. This alone moves accuracy 5–8 points.
3. Combine: Spider train (~7K) + BIRD train (~9K) → ~16K examples. **Don't use 80K — quality > quantity for QLoRA on 7B.**
4. Hold out a 500-example mixed eval set for fast iteration during training.

**Deliverable:** `data/processed/train.jsonl`, `data/processed/eval_holdout.jsonl`

**Heads up:** BIRD examples include "external knowledge" hints — decide whether to include them in your prompt or not. Document the choice. Both are valid; interviewers will ask why.

### Phase 3 — Fine-Tuning with Unsloth + QLoRA (Weeks 3–4)

**Goal:** Trained adapter that beats baseline by ≥10 points on Spider, ≥5 on BIRD.

1. Use **Unsloth** — not vanilla HF PEFT. Roughly 2× faster on Colab T4 and uses ~40% less memory. Critical for free-tier feasibility.
2. Hyperparameters that work for this scale (starting point, not gospel):
   - LoRA rank: 16, alpha: 32
   - Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
   - Learning rate: 2e-4 with cosine schedule
   - Batch size: 4, gradient accumulation: 4 (effective 16)
   - Epochs: 2 (more overfits on this size)
   - Max seq length: 2048 (schema + question + SQL fits)
3. Training time on T4: ~6–8 hours for 2 epochs on 16K examples. **Save checkpoints every 500 steps** because Colab will disconnect.
4. Track everything on W&B. Make the project public — link it from your resume.

**Deliverable:** Adapter pushed to HuggingFace Hub at `your-username/nl2sql-qwen-coder-7b-qlora`.

**Reality check on numbers:** With this setup, realistic results are:
- Spider dev execution accuracy: **70–78%** (don't promise 85%; that's full fine-tune territory)
- BIRD dev execution accuracy: **40–55%** (BIRD is genuinely hard; SOTA is ~70%)

**Report what you actually get. Lying about numbers is the fastest way to get caught in an interview.**

### Phase 4 — Self-Correcting Agent (Week 5)

**Goal:** The differentiator. Add execution feedback loop.

1. **Executor** (`src/agent/executor.py`): Takes SQL + DB connection, runs it, returns either result rows or structured error.
2. **Self-correction loop** (`src/agent/self_correction.py`):
   ```
   for attempt in range(MAX_RETRIES=2):
       sql = model.generate(prompt)
       result = executor.run(sql)
       if result.success and result.rows is not None:
           return result
       prompt = build_retry_prompt(original_question, schema, sql, result.error)
   return graceful_failure(last_attempt)
   ```
3. Re-run evaluation with the loop on. Expect **+3 to +7 points** on Spider, **+5 to +10** on BIRD (BIRD benefits more because errors are more common).
4. **Log retry rate, success-on-retry rate.** These are the metrics that matter for a real bullet point.

**Deliverable:** Comparison table — base model vs fine-tuned vs fine-tuned + agent loop.

### Phase 5 — Schema Retrieval (optional, Week 5.5 if time permits)

For DBs with many tables, embed each table's schema + description, retrieve top-5 relevant tables for the question, only include those in the prompt. This is what lets your system handle real enterprise DBs, not just the toy Spider schemas.

Skip this if running short on time. The agent loop is more impressive.

### Phase 6 — Deployment (Week 6)

**Goal:** Public, working demo a recruiter can click on.

1. **FastAPI service** (`src/api/main.py`): `POST /query` endpoint that takes `{question, db_id}`, returns `{sql, results, retries_used}`.
2. **Streamlit demo** (`app/streamlit_demo.py`): Schema picker, question input, shows generated SQL + result table + retry trace.
3. **Hosting:**
   - Streamlit demo → **HuggingFace Spaces** (free, runs the 4-bit quantized model on CPU; slow but works for demos)
   - FastAPI for resume → Dockerize, push to GitHub, document `docker run` command. Don't pay for hosting unless someone's actually using it.
4. **README badges:** Spider score, BIRD score, retry success rate, link to W&B project, link to live demo.

### Phase 7 — Write-Up & Resume (Week 7)

Write a short blog post (Medium / personal site / dev.to) — 1500 words, with charts. This 10× the value of the project for almost no extra work. Recruiters Google your name; the blog post shows up.

---

## Evaluation Methodology

Use **execution accuracy** (does running the predicted SQL on the DB give the same rows as running the gold SQL?), not exact match. Exact match is broken — there are 5 ways to write the same correct query.

```python
# src/eval/execution_accuracy.py — the only metric that matters
def execution_accuracy(pred_sql, gold_sql, db_path):
    pred_result = execute(pred_sql, db_path)
    gold_result = execute(gold_sql, db_path)
    return set(pred_result) == set(gold_result)
```

Report:
- Overall execution accuracy
- Breakdown by difficulty (easy/medium/hard/extra) — Spider provides this
- Confusion buckets: which query types fail most? (joins? aggregations? nested?)

**Error analysis is the most underrated part.** Pick 30 failures, categorize them, put the table in your README. This single table is what separates "I built a project" from "I understand my project."

---

## Resume Bullets (Template — Fill With YOUR Actual Numbers)

> Don't copy these. Run your project, get YOUR numbers, then write bullets. The structure:

- Fine-tuned **Qwen2.5-Coder-7B** with QLoRA (Unsloth) on **16K Spider+BIRD examples**, achieving **[X]% execution accuracy on Spider dev** and **[Y]% on BIRD dev** — outperforming zero-shot baseline by **[Z] points**
- Designed **self-correcting agent loop** that re-prompts the model with execution errors, improving BIRD accuracy by **[+N] points** and reducing failed queries from **[A]% to [B]%**
- Built **schema-aware retrieval pipeline** (BGE embeddings + FAISS) to handle databases with 100+ tables, keeping prompt size under 2K tokens
- Deployed via **FastAPI + Docker**, with **Streamlit demo** on HuggingFace Spaces; instrumented with **W&B** for experiment tracking ([link])

**What's missing on purpose:** "500 queries/day," "production system," "saved company $X." Don't fake these. A junior with honest, specific numbers beats a junior with made-up scale every time.

---

## Interview Prep — Questions You Will Be Asked

Have a confident, specific answer for each:

1. **Why Qwen2.5-Coder over CodeLlama or DeepSeek-Coder?** (Benchmark scores; release date; trained on more recent code)
2. **Why QLoRA instead of full fine-tuning?** (VRAM; LoRA preserves base model; faster experimentation)
3. **Why these specific LoRA hyperparameters?** (Have a reason for rank 16. "I tried 8, 16, 32 — 16 was the best speed/quality trade-off.")
4. **What's the difference between Spider and BIRD?** (BIRD: real DBs, dirtier schemas, external knowledge, harder; Spider: cleaner, more academic)
5. **Why execution accuracy and not exact match?** (You know this one)
6. **What's your biggest failure mode?** (Pull from your error analysis. Be specific. "Nested subqueries with HAVING clauses — 40% accuracy. I tried X, Y, Z." This answer is the interview.)
7. **How would you scale this to 1000 QPS?** (Quantization to GGUF, vLLM serving, prompt caching, response caching, schema retrieval to keep contexts small. Don't claim you did this — say how you would.)
8. **Why didn't you use [bigger model / fancier technique]?** ("Trade-offs. I had X compute, Y time, and the marginal gain wasn't worth it for a portfolio project.")

---

## What NOT to Do (Failure Modes I've Seen)

- **Don't fine-tune for 10 epochs.** You'll overfit hard. 2 is the sweet spot for this dataset size.
- **Don't ignore data leakage.** Some BIRD/Spider tutorials accidentally leak dev examples into train. Verify your splits.
- **Don't report only easy queries.** Spider has difficulty buckets — show your accuracy on `extra hard`. If it's 30%, say so. Honesty signals competence.
- **Don't skip the executor.** Models hallucinate column names. Execution-grounded evaluation is non-negotiable.
- **Don't deploy a 14GB model to a free server and brag about latency.** Quantize to 4-bit GGUF for inference, or use the API model for the demo and the fine-tuned one only for the benchmark numbers. Be transparent about which is which.
- **Don't pad with vanity metrics.** "Handled 500 queries/day" from a cron job is a lie that interviewers spot instantly.

---

## Stretch Goals (If You Finish Early)

- **DPO on top of SFT:** Use execution success/failure as preference signal. Real ML signal, not just a buzzword.
- **Ambiguity detection:** Train a small classifier or use the model itself to detect when a question is ambiguous and ask a clarifying question. Almost no one does this.
- **Domain adaptation:** Take your model and fine-tune the adapter on a small domain-specific dataset (e-commerce, healthcare). Show it generalizes.
- **Compare against GPT-4o / Claude on the same eval.** Be honest if they beat you — they will. The point is showing you can evaluate fairly. Cost-per-query comparison makes the bullet much sharper.

---

## References

- Spider: Yu et al., 2018 — https://arxiv.org/abs/1809.08887
- BIRD: Li et al., 2023 — https://arxiv.org/abs/2305.03111
- Qwen2.5-Coder: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- Unsloth docs: https://docs.unsloth.ai/
- QLoRA paper: Dettmers et al., 2023 — https://arxiv.org/abs/2305.14314

---

## License

MIT (or whatever you prefer). The Spider and BIRD datasets have their own licenses — read them.

---

**Last note:** Building this honestly takes 6–8 weeks of consistent work, not a weekend. If a tutorial or LinkedIn post tells you it's a 2-day project, they're either lying or building a toy. The depth is the point — that's what separates this from the 200 vanilla text-to-SQL projects on GitHub.

Build it. Get real numbers. Write the blog post. Then you have something to talk about.
