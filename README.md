# AUSLegalSearchv3 – Multi-Faceted Unified Legal AI Platform

---

## Unified System Architecture

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
    |          |           |                                              |
[FastAPI :8000]  [Gradio :7866+]  [Streamlit :8501]       [PGVector/PostgreSQL DB]
    |          |           |                                              |
    |          |           +---Intelligent Legal UI/Document Control------+         
    |          +---Modern LLM-Driven Chat & Hybrid Search UI---+          
    +---REST API, RAG, Embedding, Ingestion Control---+                  
```

### Components and Relationships

- **FastAPI backend (`:8000`)**: The unified API layer powering ingestion, advanced search (BM25, vector, hybrid), chat/RAG endpoints, and managing all data/model orchestration. All apps depend on FastAPI for business logic and DB interaction.
- **Gradio frontend (`:7866`, fallback 7867-7879)**: Provides a modern, interactive UI supporting:
    - Legal RAG Chat Assistant (vector-augmented, with custom prompt option)
    - Unified search page (BM25, vector, hybrid)
    - Ingestion, session, and document management
    - Built for researchers, legal staff, and power users
- **Streamlit frontend (`:8501`)**: Rapid document ingestion, search, and legal QA dashboard—great for demos, analytics, or non-technical users.
- **Postgres/pgvector DB**: Stores both raw documents and high-dimensional embeddings for fast semantic/hybrid search, session state, and audit logs.
- **Interoperation**: All components communicate over REST. User actions in either UI hit the FastAPI backend, which coordinates RAG, embedding, and vector retrieval.
- **Security**: All external traffic flows through load balancer/WAF at the network edge; only required ports are whitelisted.

---

## Deploying the Unified Stack

### 1. Server Prereqs & Installation

```sh
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-venv python3-pip git postgresql libpq-dev gcc unzip curl -y
```

### 2. Clone & Prepare

```sh
git clone https://github.com/shadabshaukat/auslegalsearchv3.git
cd auslegalsearchv2
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Setup DB & Env

- Install and bootstrap PostgreSQL + [pgvector](https://github.com/pgvector/pgvector).
- Create user/db as desired.
- Set required ENV variables before starting:
  ```
  export AUSLEGALSEARCH_DB_HOST=localhost
  export AUSLEGALSEARCH_DB_PORT=5432
  export AUSLEGALSEARCH_DB_USER=postgres
  export AUSLEGALSEARCH_DB_PASSWORD='YourPasswordHere'
  export AUSLEGALSEARCH_DB_NAME=postgres
  ```

### 4. Whitelist Required Ports

- On your server/cloud firewall and internal iptables, open:
  - **8000**: FastAPI (core REST API)
  - **7866-7879**: Gradio UI (always starts at 7866, auto-fails over)
  - **8501**: Streamlit UI
- Example iptables:
  ```sh
  sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
  sudo iptables -I INPUT -p tcp --dport 7866:7879 -j ACCEPT
  sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
  ```

---

### 5. Unified Launch: One-Command Startup

Run everything in one step:
```sh
bash run_legalsearch_stack.sh
```
- **This brings up all services and backgrounds them, checking/falling back on ports as needed.**
- To stop everything:
  ```
  bash stop_legalsearch_stack.sh
  ```
- Gradio is always on 7866 unless busy, then increments to 7879.
- FastAPI is always at 8000.
- Streamlit is always at 8501 (or as configured).

---

### 6. Application Overview (Unified Cross-UI Workflow)

- **Ingestion**: Add new legal material (PDF, HTML, etc.)—handled from either Gradio or Streamlit UI. Triggers chunking, embedding, and DB storage.
- **Search (BM25, vector, hybrid)**: Both UI frontends let you search by keywords, semantic similarity, or a fusion—either as direct search or as legal RAG helper for AI answers.
- **RAG Legal Chat Assistant (Gradio)**: The flagship conversational search feature. Every message runs a live vector search, then passes those chunks (and any user prompt) to the LLM for context-augmented answers.
- **Session management**: All activity is recorded in DB. Users can resume previous sessions, review history, and audit results.
- **Extensibility**: Enables future integration of new frontends (e.g. mobile or lightweight web apps) via FastAPI endpoints.

---

### 7. Security, Certificate, and Ops Notes

- All public endpoints are WAF-protected, with only the three critical ports open.
- Strong auth, roles, and audit capabilities are baked in (disable admin logins/development users ASAP for production).
- Certificates: Use Let's Encrypt (with Route53 DNS challenge or Nginx reverse proxy) as described earlier for best public deployment hygiene.

---

### 8. System Diagram

For reference, your deployed system looks like:

```
[Users]  <--HTTPS(load balancers/WAF)--->  [FastAPI:8000 | Gradio:7866+ | Streamlit:8501]  <--> [Postgres/pgvector]
           (all UIs go through FastAPI backend for core logic)
```

---

## Contact, Support, and Contribution

- Raise issues or feature requests at the [GitHub repository](https://github.com/shadabshaukat/auslegalsearchv2)
- Contributions (UI, search quality, embedding, legal workflow modules) are welcome.

---

**AUSLegalSearchv2 – Secure, unified, multi-modal legal intelligence for modern research and compliance.**
