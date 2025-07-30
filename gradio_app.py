"""
AUSLegalSearchv3 Gradio UI:
- Tabs for Hybrid Search, Vector Search, and 'RAG' (unified chatbot).
- RAG tab: Improved LLM/model selection, Top K, source/context formatting. Displays all available OCI GenAI models, even with fallback key logic.
"""

import gradio as gr
import requests
import os
import html
import json

API_ROOT = os.environ.get("AUSLEGALSEARCH_API_URL", "http://localhost:8000")
SESS = type("Session", (), {"user": None, "auth": None})()
SESS.auth = None

LEGAL_SYSTEM_PROMPT = """You are an expert Australian legal research and compliance AI assistant.
Answer strictly from the provided sources and context. Always cite the source section/citation for every statement. If you do not know the answer from the context, reply: "Not found in the provided legal documents."
When summarizing, be neutral and factual. Never invent legal advice."""

def login_fn(username, password):
    try:
        r = requests.get(f"{API_ROOT}/health", auth=(username, password), timeout=10)
        if r.ok:
            SESS.auth = (username, password)
            return gr.update(visible=False), gr.update(visible=True), f"Welcome, {username}!", ""
        else:
            return gr.update(visible=True), gr.update(visible=False), "", "Invalid login."
    except Exception:
        return gr.update(visible=True), gr.update(visible=False), "", "Invalid login."

def fetch_oci_models():
    try:
        resp = requests.get(f"{API_ROOT}/models/oci_genai", auth=SESS.auth, timeout=30)
        data = resp.json() if resp.ok else {}
        all_models = data.get("all", []) if isinstance(data, dict) else []

        # Save for troubleshooting
        try:
            with open("oci_models_debug.json", "w") as f:
                json.dump(all_models, f, indent=2, default=str)
            if all_models:
                print("First 2 Oracle model objects (debug):", json.dumps(all_models[:2], indent=2, default=str))
        except Exception as e:
            print("Warning: Could not write oci_models_debug.json", e)

        filtered = []
        for m in all_models:
            # Show all keys for each model (debug)
            print("Model keys:", list(m.keys()))
            # Extract model name and OCID using any present key
            display = (
                m.get("display_name")
                or m.get("model_name")
                or m.get("name")
                or m.get("id")
                or m.get("ocid")
                or str(m)
            )
            ocid = m.get("id") or m.get("ocid") or m.get("model_id") or ""
            # Put together as label and dropdown value
            capdesc = []
            if m.get("category"):
                capdesc.append(m["category"])
            if m.get("capabilities"):
                capdesc.extend([str(c) for c in m["capabilities"]])
            if m.get("operation_types"):
                capdesc.extend(["[ops: " + ", ".join(m["operation_types"]) + "]"])
            label = f"{display} ({', '.join(capdesc)})" if capdesc else display
            # Only list if an OCID/id is present
            if ocid:
                filtered.append((label, json.dumps(m)))
        # If nothing, show a clear UI diagnostic message instead of empty dropdown
        if not filtered:
            filtered.append(("No OCI GenAI models found. Check Oracle Console/config.", ""))
        return filtered
    except Exception as exc:
        print("OCI Models fetch error:", exc)
        return [(f"Error loading Oracle models: {exc}", "")]

def fetch_ollama_models():
    try:
        resp = requests.get(f"{API_ROOT}/models/ollama", auth=SESS.auth, timeout=15)
        data = resp.json() if resp.ok else []
        if isinstance(data, list):
            return [(m, m) for m in data]
        return []
    except Exception:
        return []

def format_context_sources(chunks, chunk_metadata):
    if not chunks:
        return "<div style='color:grey;padding:.5em'>No sources/chunks found for this question.</div>"
    out = "<div style='display:flex;flex-direction:column;gap:1em;'>"
    for idx, text in enumerate(chunks):
        meta = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
        citation = meta.get("citation") or meta.get("url") or "?"
        url = meta.get("url")
        url_part = f'<a href="{url}" target="_blank">{html.escape(str(citation))}</a>' if url else html.escape(str(citation))
        t = html.escape(text[:700])
        out += f"""
        <div style='border-radius:9px;border:1.5px solid #ececec;padding:10px 13px;background:#fafdff;margin-bottom:4px;font-size:1em'>
            <div style='font-weight:bold;margin-bottom:.1em'>{url_part}</div>
            <div style='margin: 6px 0 4px 0;line-height:1.54;'>{t}{'...' if len(text)>700 else ''}</div>
        </div>
        """
    out += "</div>"
    return out

def hybrid_search_fn(query, top_k, alpha):
    reranker_model = "mxbai-rerank-xsmall"
    try:
        resp = requests.post(
            f"{API_ROOT}/search/hybrid",
            json={"query": query, "top_k": top_k, "alpha": alpha, "reranker": reranker_model}, auth=SESS.auth, timeout=20
        )
        resp.raise_for_status()
        hits = resp.json()
    except Exception:
        hits = []
    return format_context_sources([h.get("text","") for h in hits], [h.get("chunk_metadata") or {} for h in hits])

def vector_search_fn(query, top_k):
    try:
        resp = requests.post(
            f"{API_ROOT}/search/vector",
            json={"query": query, "top_k": top_k}, auth=SESS.auth, timeout=20
        )
        resp.raise_for_status()
        hits = resp.json()
    except Exception:
        hits = []
    out = "<ul>"
    for idx, h in enumerate(hits, 1):
        out += f"<li>{h.get('text','')[:150]}{'...' if len(h.get('text',''))>150 else ''}</li>"
    out += "</ul>"
    return out

def rag_chatbot(question, llm_source, ollama_model, oci_model_info_json, rag_top_k):
    if not question:
        return "", "", ""
    # Step 1: Hybrid search
    try:
        resp = requests.post(
            f"{API_ROOT}/search/hybrid",
            json={"query": question, "top_k": rag_top_k, "alpha": 0.5, "reranker": "mxbai-rerank-xsmall"}, 
            auth=SESS.auth, timeout=30
        )
        resp.raise_for_status()
        hits = resp.json()
        context_chunks = [h.get("text", "") for h in hits]
        sources = [h.get("citation") or ((h.get("chunk_metadata") or {}).get("url") if (h.get("chunk_metadata") or {}) else "?") for h in hits]
        chunk_metadata = [h.get("chunk_metadata") or {} for h in hits]
    except Exception:
        context_chunks, sources, chunk_metadata = [], [], []
    answer, srcs_md = "", ""
    if llm_source == "OCI GenAI":
        oci_region = os.environ.get("OCI_REGION", "ap-sydney-1")
        try:
            model_info = json.loads(oci_model_info_json) if oci_model_info_json else {}
        except Exception:
            model_info = {}
        model_id = model_info.get("id") or model_info.get("ocid") or model_info.get("model_id") or os.environ.get("OCI_GENAI_MODEL_OCID", "")
        oci_payload = {
            "model_info": model_info,
            "oci_config": {
                "compartment_id": os.environ.get("OCI_COMPARTMENT_OCID", ""),
                "model_id": model_id,
                "region": oci_region
            },
            "question": question,
            "context_chunks": context_chunks or [],
            "sources": sources or [],
            "chunk_metadata": chunk_metadata or [],
            "custom_prompt": LEGAL_SYSTEM_PROMPT,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 1024,
            "repeat_penalty": 1.1
        }
        try:
            r_oci = requests.post(f"{API_ROOT}/search/oci_rag", json=oci_payload, auth=SESS.auth, timeout=50)
            oci_data = r_oci.json() if r_oci.ok else {}
            answer = oci_data.get("answer", "")
            if answer and "does not support TextGeneration" in answer:
                answer += "<br><span style='color:#c42;font-size:1.03em;'>This OCI model does not support text generation. Make sure you select a model marked as LLM/TextGeneration and check Oracle Console.</span>"
        except Exception as e:
            answer = f"Error querying OCI GenAI: {e}"
    else:
        rag_payload = {
            "question": question,
            "context_chunks": context_chunks or [],
            "sources": sources or [],
            "chunk_metadata": chunk_metadata or [],
            "custom_prompt": LEGAL_SYSTEM_PROMPT,
            "model": ollama_model or "llama3",
            "reranker_model": "mxbai-rerank-xsmall",
            "top_k": rag_top_k
        }
        try:
            r = requests.post(f"{API_ROOT}/search/rag", json=rag_payload, auth=SESS.auth, timeout=35)
            rag_data = r.json() if r.ok else {}
            answer = rag_data.get("answer", "")
        except Exception as e:
            answer = f"Error querying Ollama: {e}"
    answer_html = f"<div style='color:#10890b;font-size:1.1em;font-family:Menlo,Monaco,monospace;margin-top:0.7em;white-space:pre-wrap'>{answer or '[No answer returned]'}"
    answer_html += "</div>"
    context_html = format_context_sources(context_chunks, chunk_metadata)
    return answer_html, context_html, sources

with gr.Blocks(title="AUSLegalSearch RAG UI", css="""
#llm-answer-box {
    color: #10890b !important;
    font-size: 1.13em;
    font-family: Menlo, Monaco, 'SFMono-Regular', monospace;
    background: #f2fff4 !important;
    border-radius: 7px;
    border: 2px solid #c5ebd3;
    margin-bottom:12px;
    min-height: 32px;
}
""") as demo:
    gr.Markdown("# AUSLegalSearch RAG Platform")

    login_box = gr.Row(visible=True)
    with login_box:
        gr.Markdown("## Login to continue")
        username = gr.Textbox(label="Username", value="legal_api")
        password = gr.Textbox(label="Password", type="password")
        login_err = gr.Markdown("")
        login_btn = gr.Button("Login")

    with gr.Row(visible=False) as app_panel:
        with gr.Tabs():
            with gr.Tab("Hybrid Search"):
                hybrid_query = gr.Textbox(label="Enter a legal research question", lines=2)
                hybrid_top_k = gr.Number(label="Top K Results", value=10, precision=0)
                hybrid_alpha = gr.Slider(label="Hybrid weighting (semantic/keyword)", value=0.5, minimum=0.0, maximum=1.0)
                hybrid_btn = gr.Button("Hybrid Search")
                hybrid_results = gr.HTML(label="Results", value="", show_label=False)
                hybrid_btn.click(
                    hybrid_search_fn,
                    inputs=[hybrid_query, hybrid_top_k, hybrid_alpha],
                    outputs=[hybrid_results]
                )
            with gr.Tab("Vector Search"):
                vector_query = gr.Textbox(label="Vector Search Query", lines=2)
                vector_top_k = gr.Number(label="Top K Results", value=10, precision=0)
                vector_btn = gr.Button("Run Vector Search")
                vector_results = gr.HTML(label="Result cards", value="", show_label=False)
                vector_btn.click(
                    vector_search_fn,
                    inputs=[vector_query, vector_top_k],
                    outputs=[vector_results]
                )
            with gr.Tab("RAG"):
                gr.Markdown("#### RAG-Powered Legal Chat")
                llm_source = gr.Dropdown(label="LLM Source", choices=["Local Ollama", "OCI GenAI"], value="Local Ollama")
                ollama_model = gr.Dropdown(label="Ollama Model", choices=[], visible=True)
                oci_model = gr.Dropdown(label="OCI GenAI Model", choices=[], visible=False)
                rag_top_k = gr.Number(label="Top K Context Chunks", value=10, precision=0)
                def update_model_dropdowns(src):
                    if src == "OCI GenAI":
                        return gr.update(visible=False), gr.update(choices=fetch_oci_models(), visible=True)
                    else:
                        return gr.update(choices=fetch_ollama_models(), visible=True), gr.update(choices=[], visible=False)
                llm_source.change(update_model_dropdowns, inputs=[llm_source], outputs=[ollama_model, oci_model])
                question = gr.Textbox(label="Enter your legal or compliance question", lines=2)
                ask_btn = gr.Button("Ask")
                answer = gr.HTML(label="Answer", elem_id="llm-answer-box", value="")
                context = gr.HTML(label="Context / Sources", value="", show_label=False)
                ask_btn.click(
                    rag_chatbot,
                    inputs=[question, llm_source, ollama_model, oci_model, rag_top_k],
                    outputs=[answer, context, gr.State()]
                )

    login_btn.click(
        login_fn,
        inputs=[username, password],
        outputs=[
            login_box,
            app_panel,
            login_err,
            login_err,
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866)
