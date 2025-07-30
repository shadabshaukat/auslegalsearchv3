# AUSLegalSearchv3 – Multi-Faceted Unified Legal AI Platform (Ollama ⬄ OCI GenAI ⬄ Oracle 23ai)

---

## Unified System Architecture (v3)

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
    |          |           +---Modern LLM/Cloud UI (Gradio tabs: Hybrid, Chat, OCI GenAI)----+ 
    |          +---Multisource LLM-driven Chat, Hybrid Search, OCI GenAI---+                 
    +---REST API: ingestion, retrieval, RAG (Ollama and OCI), DB bridge----+                 
```

### Key v3 Capabilities

- **Select LLM Source:** Use either local Ollama models _or_ Oracle Cloud GenAI for RAG/Augment/Chat via a unified dropdown in the Gradio UI.
- **Dedicated GenAI Tab:** Configure OCI credentials (Compartment OCID, Model OCID, Region) and run direct Oracle GenAI QA/RAG queries interactively.
- **Secondary DB Connectivity:** Query Oracle 23ai databases through the FastAPI backend for advanced legal/compliance/analytics flows.
- **Legacy Support:** All local features and hybrid search underpinnings run unmodified if you choose only Ollama/backed search.
- **Unified APIs:** All app UIs (Gradio, Streamlit) and future automation connect through a robust FastAPI foundation.

---

## Deployment Instructions (v3)

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
> The requirements.txt now installs [oci](https://pypi.org/project/oci/) for Oracle Cloud GenAI support and [oracledb](https://pypi.org/project/oracledb/) for Oracle DB connectivity.

### 3. Environment Variables

**Core Database Variables:**

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

**Oracle Cloud GenAI (set if using OCI features):**

```
export OCI_USER_OCID='ocid1.user.oc1...'
export OCI_TENANCY_OCID='ocid1.tenancy.oc1...'
export OCI_KEY_FILE='/path/to/oci_api_key.pem'
export OCI_KEY_FINGERPRINT='xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx:xx'
export OCI_REGION='ap-sydney-1'
export OCI_COMPARTMENT_OCID='ocid1.compartment.oc1...'
export OCI_GENAI_MODEL_OCID='ocid1.generativeaiocid...'
```
- These are required for the "OCI GenAI" tab and any RAG/QA using Oracle GenAI.
- You can also set these interactively from the Gradio UI for session-only usage.

**Oracle 23ai DB integration (optional, for DB tab/api usage):**
```
export ORACLE_DB_USER='your_db_user'
export ORACLE_DB_PASSWORD='your_db_password'
export ORACLE_DB_DSN='your_db_high'
export ORACLE_WALLET_LOCATION='/path/to/wallet/dir' # (optional, only for Autonomous DBs with wallet)
```
- These can also be provided per-request in the UI.

---

### 4. Database/Firewall/Network Prep

- Install and initialize **PostgreSQL** and **pgvector** as usual for core legal search functions.
- Create users, DB, etc.
- For Oracle 23ai features, provision and create connection string/wallet as directed by Oracle cloud documentation.
- Open/forward these ports:
  - 8000 (FastAPI)
  - 7866-7879 (Gradio UI)
  - 8501 (Streamlit UI)

Example (for local/VM firewall):
```sh
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 7866:7879 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
```

---

### 5. Starting and Stopping the Stack

Launch all services:
```sh
bash run_legalsearch_stack.sh
```
Stop all services:
```sh
bash stop_legalsearch_stack.sh
```
- Gradio: http://localhost:7866
- FastAPI: http://localhost:8000
- Streamlit: http://localhost:8501

> Default startup script configurations remain unchanged from previous versions.

---

## Application Workflow (v3)

- **Ingestion**: Upload legal docs to index via UI—handled exactly as in v2.
- **Search (BM25, Vector, Hybrid)**: Choose your retrieval method from both Gradio and Streamlit. Hybrid and Chat tabs allow for LLM source toggle (Ollama vs. OCI GenAI).
- **RAG-Enabled Chat (Gradio)**: All chats/searches can use either local LLM models (Ollama, etc.) or call out to Oracle Cloud GenAI—toggle using the "LLM Source" dropdown.
- **GenAI Tab**: Manage and test Oracle Cloud GenAI setup directly; set credentials and region once per session, run QA interactively, debug config.
- **Oracle 23ai DB Integration**: Use the backend `/db/oracle23ai_query` endpoint for advanced legal/analytics DB queries, or integrate from future UI upgrades.

---

## Security, Certificates, and Operational Notes

- All network endpoints should be fronted by a WAF, load balancer, or HTTPS reverse proxy.
- **Never** commit or expose your OCI/Oracle credentials; always use environment variables and/or mounted secrets!
- Rotate/limit credentials as required by organizational policy.
- Use Let's Encrypt for certificate management as described in previous documentation.

---

## Contact and Contribution

Raise issues or PRs for v3 at:  
[https://github.com/shadabshaukat/auslegalsearchv3](https://github.com/shadabshaukat/auslegalsearchv3)

---

**AUSLegalSearchv3 — Secure, unified, multi-cloud, multi-model legal AI search and analytics for the modern firm.**
