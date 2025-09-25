# AUSLegalSearchv3 — Agentic Multi-Faceted Legal AI Platform (Ollama ⬄ OCI GenAI ⬄ Oracle 23ai)

---

## 🚀 New in v3: Agentic RAG with Chain-of-Thought Reasoning

**AUSLegalSearchv3 now includes Agentic Chain-of-Thought RAG — a world-class, explainable, multi-step reasoning engine for legal and compliance practitioners.**

- **Agentic Chat Tab:** Natural language questions yield stepwise, structured reasoning (Chain-of-Thought, CoT). Each agentic answer details "Thought", "Action", "Evidence", "Reasoning", and "Conclusion" steps—driving trustworthy, auditable outcomes.
- **Interactive UI:** Each CoT answer is shown with visual timeline cards and clear source citations, for legal-grade explainability.
- **Seamless Model Switching:** Instantly toggle between local Ollama LLMs and Oracle Cloud GenAI for agentic RAG.
- **Professional User Experience:** All tabs present clear progress/wait-state animations, robust error handling, and a big-tech–grade, modern UX.
- **Reproducible Reasoning:** Every answer’s “chain” is preserved for transparency, audit, or retraining.

### Example: UI Chain-of-Thought Flow

![Example: Chain-of-Thought UI](agentic_cot_ui.png)

---

## Architecture and Workflow

<img width="2548" height="2114" alt="AusLegalSearchv3" src="https://github.com/user-attachments/assets/d90a5d18-d769-44b9-a6b8-5e75ba47727c" />


### System Overview

```
                                              [ Route53 (DNS) ]
                                                      |
                                      auslegal.oraclecloudserver.com (or custom domain)
                                                      |
                                  +--------------------+--------------------+
                                  |                                         |
                          [OCI Public Load Balancer]                [Nginx (public VM, alt)]
                                  |                                         |
                                   -------- WAF (Web Application Firewall) ---
                                  |
                          [Backend on Ubuntu VM]  (Private IP: e.g. 10.150.1.82)
                                  |
    +----------+-----------+----------+----------------------------------------+
    |          |           |                                  |          |
[FastAPI :8000]  [Gradio :7866+]  [Streamlit :8501]    [PGVector/PostgreSQL] [Oracle 23ai DB]
    |          |           |                                  |          |
    |          |           +---Modern LLM/Cloud UI (Gradio tabs: Hybrid, Chat, OCI GenAI, Agentic)----+ 
    |          +---Multisource LLM-driven Chat, Hybrid Search, OCI GenAI---+                 
    +---REST API: ingestion, retrieval, RAG (Ollama and OCI), DB bridge, agentic reasoning----+                 
```

### Agentic RAG Chain-of-Thought Workflow

<img width="2000" height="12600" alt="image" src="https://github.com/user-attachments/assets/05148d8c-1327-47da-99ee-5c4e6421e7f8" />


---

## Key Capabilities

- **Agentic Chain-of-Thought RAG:** Legal query explanations as auditable, step-wise deductive timelines—unmatched for risk, compliance, research traceability.
- **LLM Source Diversity:** Run RAG and Agentic queries using either Ollama or Oracle Cloud GenAI—one switch, zero config hassle.
- **GenAI QA & RAG Tab:** Configure Oracle GenAI, see live status, and benchmark against local models instantly.
- **Hybrid / Vector / BM25 Retrieval:** Pick search mode as needed; results are always shown with context, sources, and explainability.
- **Plug-and-Play Backend:** All services—REST API, UI, Oracle DB, and PGVector—run together with clear network separation.

---

## Deployment Instructions

### 1. System Prerequisites

```sh
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-venv python3-pip git postgresql libpq-dev gcc unzip curl -y
```

### 2. Clone, Prepare, and Install Dependencies

```sh
git clone https://github.com/shadabshaukat/auslegalsearchv3.git
cd auslegalsearchv3
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
> Requirements include [oci](https://pypi.org/project/oci/) and [oracledb](https://pypi.org/project/oracledb/) for full GenAI and DB coverage.

### 3. Environment Variables (v3+)

**Database Setup:**
```
export AUSLEGALSEARCH_DB_HOST=localhost
export AUSLEGALSEARCH_DB_PORT=5432
export AUSLEGALSEARCH_DB_USER=postgres
export AUSLEGALSEARCH_DB_PASSWORD='YourPasswordHere'
export AUSLEGALSEARCH_DB_NAME=postgres
```

**API/Backend:**
```
export FASTAPI_API_USER=legal_api
export FASTAPI_API_PASS=letmein
export AUSLEGALSEARCH_API_URL=http://localhost:8000
```

**Oracle Cloud GenAI:**
```
export OCI_USER_OCID='ocid1.user.oc1...'
export OCI_TENANCY_OCID='ocid1.tenancy.oc1...'
export OCI_KEY_FILE='/path/to/oci_api_key.pem'
export OCI_KEY_FINGERPRINT='xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx'
export OCI_REGION='ap-sydney-1'
export OCI_COMPARTMENT_OCID='ocid1.compartment.oc1...'
export OCI_GENAI_MODEL_OCID='ocid1.generativeaiocid...'
```

**Oracle 23ai DB integration (optional):**
```
export ORACLE_DB_USER='your_db_user'
export ORACLE_DB_PASSWORD='your_db_password'
export ORACLE_DB_DSN='your_db_high'
export ORACLE_WALLET_LOCATION='/path/to/wallet/dir'
```

---

### 4. Network & Database Setup

- PostgreSQL, PGVector, and any Oracle DBs as per your needs.
- Open for: 8000 (FastAPI), 7866–7879 (Gradio), 8501 (Streamlit).
- Example:
```sh
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 7866:7879 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
```

---

### 5. Launch

```sh
bash run_legalsearch_stack.sh
# To stop: bash stop_legalsearch_stack.sh
# Gradio:   http://localhost:7866
# FastAPI:  http://localhost:8000
# Streamlit http://localhost:8501
```

---

## API Specification

### `/chat/agentic` — Agentic Chain-of-Thought LLM Endpoint

Submit a legal or compliance question and receive a structured chain-of-thought reasoning trace with cited sources.

**POST** `/chat/agentic`
#### Request
```json
{
  "llm_source": "ollama",           // or "oci_genai"
  "model": "llama3" | "orca-mini" | ...,
  "message": "Explain the procedure for contesting a will in NSW.",
  "chat_history": [],
  "system_prompt": "...",
  "temperature": 0.15,
  "top_p": 0.9,
  "max_tokens": 1920,
  "repeat_penalty": 1.07,
  "top_k": 12,
  "oci_config": {
    "compartment_id": "...", "model_id": "...", "region": "ap-sydney-1"
  }
}
```

#### Response
```json
{
  "answer": "Step 1 - Thought: ...\nStep 2 - Action: ...\nFinal Conclusion: ...", // Markdown Chain-of-Thought trace
  "sources": [
    "Succession Act 2006 (NSW) s 96", "Practical Law Practice Note", "Case: Smith v Doe [2022] NSWSC 789"
  ],
  "chunk_metadata": [
    {"citation": "Succession Act 2006 (NSW) s 96", "url": "https://legislation.nsw.gov.au/view/html/inforce/current/act-2006-080#sec.96"},
    ...
  ],
  "context_chunks": [
    "Section 96 of the Act provides that...",
    "The general process in New South Wales is..."
  ]
}
```

**Notes:**
- Each CoT step is prefixed as "Step X - [Label]:" for consistent downstream parsing/display.
- Sources/citations and context are matched to each reasoning step for legal traceability.
- Error handling will yield string "answer" fields with diagnostics on failure.

---

## Application Workflow

- **Ingestion**: Document upload and indexing for efficient search and legal analytics.
- **Search (BM25, Vector, Hybrid)**: Choice of retrieval for flexibility and accuracy.
- **Agentic Chat/CoT**: Every question yields a detailed, structured reasoning path, backed by cited context.
- **GenAI & Oracle DBs**: Integrated, no-fuss access and configuration from the UI tabs.

---

## Security and Operations

- Always secure endpoints behind WAF, firewall, or proxy.
- Store all credentials OUTSIDE your repo, in environment or secret managers.
- Use Let's Encrypt for TLS by default.
- Rotating secrets and policy compliance is essential for enterprise rollout.

---

## Contribution and Support

Raise issues, feature requests, or PRs at:  
[https://github.com/shadabshaukat/auslegalsearchv3](https://github.com/shadabshaukat/auslegalsearchv3)

---

**AUSLegalSearchv3 — Enterprise-grade, agentic, explainable legal AI built for the modern legal practice.**
