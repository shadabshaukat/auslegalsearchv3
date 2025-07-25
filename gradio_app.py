"""
Gradio frontend for AUSLegalSearchv3:
- Default Top K is now 10 for Hybrid, Vector, Chat retrievals.
- Chat Tab exposes controls for temperature, top_p, max_tokens, repeat_penalty for customizing the LLM's behavior. These are sent to the FastAPI backend for each response.
"""

import gradio as gr
import requests
import os

API_ROOT = os.environ.get("AUSLEGALSEARCH_API_URL", "http://localhost:8000")

class APISession:
    def __init__(self):
        self.user = None
        self.auth = None
        self.history = []

    def login(self, username, password):
        try:
            r = requests.get(f"{API_ROOT}/health", auth=(username, password), timeout=10)
            if r.ok:
                self.auth = (username, password)
                self.user = username
                return True
        except Exception:
            pass
        return False

SESS = APISession()

LEGAL_SYSTEM_PROMPT = """You are an expert Australian legal research and compliance AI assistant.
Answer strictly from the provided sources and context. Always cite the source section/citation for every statement. If you do not know the answer from the context, reply: "Not found in the provided legal documents."
When summarizing, be neutral and factual. Never invent legal advice."""

def fetch_llm_and_reranker_choices():
    llms, rerankers = [], []
    try:
        r = requests.get(f"{API_ROOT}/models/ollama", auth=SESS.auth, timeout=10)
        if r.ok and isinstance(r.json(), list):
            llms = r.json()
    except Exception:
        pass
    try:
        r = requests.get(f"{API_ROOT}/models/rerankers", auth=SESS.auth, timeout=10)
        if r.ok and isinstance(r.json(), list):
            rerankers = r.json()
    except Exception:
        pass
    return llms, rerankers

def login_fn(username, password):
    login_ok = SESS.login(username, password)
    if not login_ok:
        return (
            gr.update(visible=True), gr.update(visible=False), "",
            "Invalid login.",
            gr.update(choices=[], value="", allow_custom_value=True),
            gr.update(choices=[], value="", allow_custom_value=True),
            gr.update(choices=[], value="", allow_custom_value=True),
            gr.update(choices=[], value="", allow_custom_value=True),
            gr.update(choices=[], value="", allow_custom_value=True),
            gr.update(choices=[], value="", allow_custom_value=True)
        )
    llm_choices, reranker_choices = fetch_llm_and_reranker_choices()
    default_llm = llm_choices[0] if llm_choices else ""
    default_reranker = reranker_choices[0] if reranker_choices else ""
    dropdown_out = gr.update(choices=llm_choices, value=default_llm, allow_custom_value=True)
    reranker_out = gr.update(choices=reranker_choices, value=default_reranker, allow_custom_value=True)
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        f"Welcome, {username}!",
        "",
        dropdown_out,          # hybrid_llm_model
        reranker_out,          # hybrid_reranker_model
        dropdown_out,          # vector_llm_model
        reranker_out,          # vector_reranker_model
        dropdown_out,          # chat_llm_model
        reranker_out           # chat_reranker_model
    )

def format_metadata(meta: dict):
    if not meta:
        return ""
    out = ["<div style='padding:1px 0 6px 0'><b>Metadata:</b><ul style='margin:4px 0 4px 14px;padding:0;'>"]
    for k, v in meta.items():
        if k == "url":
            continue
        out.append(f"<li style='font-size:.98em;'><span style='color:#476088;'>{k}:</span> <code style='color:#347;'>"
                   f"{str(v)}</code></li>")
    out.append("</ul></div>")
    return "".join(out)

def format_hybrid_results(hits):
    if not hits:
        return "No results found."
    out = "<div style='display:flex;flex-direction:column;gap:1em;'>"
    for idx, h in enumerate(hits, 1):
        meta = h.get("chunk_metadata") or {}
        citation = h.get("citation") or (meta.get("url") if meta and meta.get("url") else "?")
        url = meta.get("url")
        url_part = f'<a href="{url}" target="_blank">{citation}</a>' if url else citation
        score = f"{h.get('hybrid_score', 0):.3f}"
        text = h.get("text", "")[:700]
        out += f"""
        <div style='border-radius:9px;border:1.5px solid #ececec;padding:14px 18px;background:#fafdff;margin-bottom:4px;'>
            <div style='font-weight:bold;margin-bottom:.1em'>{url_part}</div>
            <div style='color:#34758d;font-size:.92em;'>Confidence: <b>{score}</b></div>
            <div style='margin: 8px 0 7px 0;line-height:1.56;'>{text}{'...' if len(h.get('text', ''))>700 else ''}</div>
            {format_metadata(meta)}
        </div>
        """
    out += "</div>"
    return out

def hybrid_search_fn(query, top_k, alpha, prompt, llm_model, reranker_model):
    processing_html = "<div style='color:#888;font-style:italic;margin:.7em 0;'>Processing…</div>"
    table_md, answer_html, sources_md = "", processing_html, ""
    try:
        resp = requests.post(f"{API_ROOT}/search/hybrid", json={
            "query": query, "top_k": int(top_k), "alpha": float(alpha), "reranker": reranker_model
        }, timeout=30, auth=SESS.auth)
        resp.raise_for_status()
        hits = resp.json()
    except Exception:
        return format_hybrid_results([]), "", ""
    if not hits:
        return "No results found for this query.", "", ""
    table_md = format_hybrid_results(hits)
    context_chunks = [h["text"] for h in hits]
    citation_list = [h.get("citation") or ((h.get("chunk_metadata") or {}).get("url") if (h.get("chunk_metadata") or {}) else "?") for h in hits]
    try:
        rag_payload = {
            "question": query,
            "context_chunks": context_chunks,
            "sources": citation_list,
            "custom_prompt": prompt,
            "model": llm_model,
            "reranker_model": reranker_model,
            "top_k": int(top_k)
        }
        r_rag = requests.post(f"{API_ROOT}/search/rag", json=rag_payload, auth=SESS.auth)
        rag_data = r_rag.json() if r_rag.ok else {}
        answer = rag_data.get("answer", "")
    except Exception:
        answer = ""
    sources_md = ""
    if citation_list:
        sources_boxes = " ".join([
            f"<span style='border-radius:6px;background:#e7f1fe;color:#234;font-size:.93em;padding:4px 12px 4px 12px;margin:5px 5px 5px 0;display:inline-block;'>{c}</span>"
            for c in citation_list
        ])
        sources_md = f"<div style='margin-top:7px;margin-bottom:4px;'>Sources: {sources_boxes}</div>"
    answer_html = f"<div style='color:#10890b;font-size:1.1em;font-family:Menlo,Monaco,monospace;margin-top:0.5em;white-space:pre-wrap'>{answer}</div>" if answer else ""
    return table_md, answer_html, sources_md

def format_vector_results(hits):
    if not hits:
        return "No vector results found."
    out = "<div style='display:flex;flex-direction:column;gap:1em;'>"
    for h in hits:
        meta = h.get("chunk_metadata") or {}
        url = meta.get("url")
        citation = h.get("citation") or (url if url else "?")
        url_part = f'<a href="{url}" target="_blank">{citation}</a>' if url else citation
        score = h.get("score", 0)
        text_short = h.get("text", "")[:650]
        out += f"""
        <div style='border-radius:8px;border:1.2px solid #ecedf0;padding:11px 16px;background:#fcfdfe;margin-bottom:3px;'>
            <div style='font-weight:bold;'>{url_part}</div>
            <div style='color:#3a7096;font-size:.91em;'>Score: <b>{score:.3f}</b></div>
            <div style='margin: 8px 0 5px 0;line-height:1.51;'>{text_short}{'...' if len(h.get('text', ''))>650 else ''}</div>
            {format_metadata(meta)}
        </div>
        """
    out += "</div>"
    return out

def vector_search_fn(query, top_k, reranker_model, llm_model):
    try:
        resp = requests.post(f"{API_ROOT}/search/vector", json={
            "query": query, "top_k": int(top_k), "reranker": reranker_model, "llm_model": llm_model
        }, timeout=30, auth=SESS.auth)
        resp.raise_for_status()
        hits = resp.json()
    except Exception as e:
        return "Vector search error: %s" % e
    return format_vector_results(hits)

def gradio_chat_history_from_history(history):
    chat_msgs = []
    for item in history or []:
        if "role" in item and "content" in item and item["role"] in ("user","assistant"):
            chat_msgs.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, list) and len(item)==2:
            chat_msgs.append({"role": "user", "content": item[0]})
            chat_msgs.append({"role": "assistant", "content": item[1]})
    return chat_msgs

def backend_chat_history_from_history(history):
    chat_msgs = []
    for item in history or []:
        if "role" in item and "content" in item and item["role"] in ("user","assistant"):
            chat_msgs.append({"role": item["role"], "content": item["content"]})
        elif isinstance(item, list) and len(item)==2:
            chat_msgs.append({"role": "user", "content": item[0]})
            chat_msgs.append({"role": "assistant", "content": item[1]})
    return chat_msgs

def chat_tab_fn(message, history, chat_top_k, temperature, top_p, max_tokens, repeat_penalty, llm_model, reranker_model):
    N = 2
    history_qa = []
    tmp = []
    for msg in backend_chat_history_from_history(history):
        if msg["role"] == "user":
            tmp = [msg["content"]]
        elif msg["role"] == "assistant" and tmp:
            tmp.append(msg["content"])
            history_qa.append(tmp)
            tmp = []
    last_qa_chunks = []
    for q, a in history_qa[-N:]:
        last_qa_chunks.append(f"User: {q}\nAssistant: {a}")
    try:
        search_resp = requests.post(f"{API_ROOT}/search/hybrid", json={
            "query": message, "top_k": int(chat_top_k), "alpha": 0.5, "reranker": reranker_model
        }, auth=SESS.auth, timeout=20)
        search_resp.raise_for_status()
        hits = search_resp.json()
        top_chunks = [h.get("text", "") for h in hits]
        sources = [h.get("citation") or ((h.get("chunk_metadata") or {}).get("url") if (h.get("chunk_metadata") or {}) else "?") for h in hits]
        metas = [h.get("chunk_metadata") or {} for h in hits]
    except Exception:
        top_chunks, sources, metas = [], [], []
    final_context = last_qa_chunks + top_chunks
    backend_history = backend_chat_history_from_history(history)
    rag_payload = {
        "question": message,
        "context_chunks": final_context,
        "sources": sources,
        "chunk_metadata": metas,
        "chat_history": backend_history,
        "model": llm_model,
        "reranker_model": reranker_model,
        "custom_prompt": LEGAL_SYSTEM_PROMPT,
        "top_k": int(chat_top_k),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "repeat_penalty": float(repeat_penalty)
    }
    try:
        resp = requests.post(f"{API_ROOT}/search/rag", json=rag_payload, auth=SESS.auth)
        if resp.ok:
            answer = resp.json().get("answer", "")
            new_hist = gradio_chat_history_from_history(history) + [{"role":"user","content":message},{"role":"assistant","content":answer}]
            return new_hist, "", gr.update(value="")
        else:
            new_hist = gradio_chat_history_from_history(history) + [{"role":"user","content":message}]
            return new_hist, f"Chat error: {resp.text}", gr.update(value="")
    except Exception as e:
        new_hist = gradio_chat_history_from_history(history) + [{"role":"user","content":message}]
        return new_hist, f"Chat error: {e}", gr.update(value="")

with gr.Blocks(title="AUSLegalSearch Gradio Minimalist UX", css="""
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
    gr.Markdown("# AUSLegalSearch Gradio Minimalist Search & Chat")

    login_box = gr.Row(visible=True)
    with login_box:
        gr.Markdown("## Login to continue")
        username = gr.Textbox(label="Username", value="legal_api")
        password = gr.Textbox(label="Password", type="password")
        login_err = gr.Markdown("")
        login_btn = gr.Button("Login")

    app_panel = gr.Row(visible=False)
    with app_panel:
        with gr.Tabs():
            with gr.Tab("Hybrid Search"):
                hybrid_query = gr.Textbox(label="Enter a legal research question", lines=2)
                hybrid_top_k = gr.Number(label="Top K Results", value=10, precision=0)
                hybrid_alpha = gr.Slider(label="Hybrid weighting (semantic/keyword)", value=0.5, minimum=0.0, maximum=1.0)
                hybrid_system_prompt = gr.Textbox(label="System Prompt", value=LEGAL_SYSTEM_PROMPT, lines=3)
                hybrid_llm_model = gr.Dropdown(label="LLM model", choices=[], value="", allow_custom_value=True)
                hybrid_reranker_model = gr.Dropdown(label="Reranker Model", choices=[], value="", allow_custom_value=True)
                hybrid_btn = gr.Button("Hybrid Search & Legal RAG")
                hybrid_results = gr.HTML(label="Results", value="", show_label=False)
                hybrid_answer = gr.HTML(label="LLM RAG Answer", elem_id="llm-answer-box", value="")
                hybrid_sources = gr.HTML(label="Citations", value="", show_label=False)
                hybrid_btn.click(
                    hybrid_search_fn,
                    inputs=[hybrid_query, hybrid_top_k, hybrid_alpha, hybrid_system_prompt, hybrid_llm_model, hybrid_reranker_model],
                    outputs=[hybrid_results, hybrid_answer, hybrid_sources]
                )

            with gr.Tab("Vector Search"):
                vector_query = gr.Textbox(label="Vector Search Query", lines=2)
                vector_top_k = gr.Number(label="Top K Results", value=10, precision=0)
                vector_llm_model = gr.Dropdown(label="LLM model", choices=[], value="", allow_custom_value=True)
                vector_reranker_model = gr.Dropdown(label="Reranker Model", choices=[], value="", allow_custom_value=True)
                vector_btn = gr.Button("Run Vector Search")
                vector_results = gr.HTML(label="Result cards", value="", show_label=False)
                vector_btn.click(
                    vector_search_fn,
                    inputs=[vector_query, vector_top_k, vector_reranker_model, vector_llm_model],
                    outputs=[vector_results]
                )

            with gr.Tab("Chat"):
                chat_llm_model = gr.Dropdown(label="LLM model", choices=[], value="", allow_custom_value=True)
                chat_reranker_model = gr.Dropdown(label="Reranker Model", choices=[], value="", allow_custom_value=True)
                chat_top_k = gr.Number(label="Top K Chunks", value=10, precision=0)
                temperature = gr.Slider(label="Temperature", value=0.1, minimum=0, maximum=1, step=0.01)
                top_p = gr.Slider(label="Top-p", value=0.95, minimum=0, maximum=1, step=0.01)
                max_tokens = gr.Number(label="Max Tokens", value=1024, precision=0)
                repeat_penalty = gr.Slider(label="Repeat Penalty", value=1.1, minimum=0.8, maximum=2.0, step=0.01)
                chatbot = gr.Chatbot(label="Legal Chat – minimalist", type="messages", show_label=False)
                chat_input = gr.Textbox(show_label=False, placeholder="Ask about a legal topic...", lines=2)
                send_btn = gr.Button("Send")
                chat_citations = gr.HTML(label="Sources", value="", show_label=False)
                def chat_wrapper(user_msg, history, chat_top_k, temperature, top_p, max_tokens, repeat_penalty, chat_llm_model, chat_reranker_model):
                    updated_hist, chat_srcs, clear_input = chat_tab_fn(
                        user_msg, history, chat_top_k, temperature, top_p, max_tokens, repeat_penalty, chat_llm_model, chat_reranker_model)
                    return updated_hist, chat_srcs, clear_input
                chat_state = gr.State([])
                send_btn.click(
                    chat_wrapper,
                    inputs=[chat_input, chatbot, chat_top_k, temperature, top_p, max_tokens, repeat_penalty, chat_llm_model, chat_reranker_model],
                    outputs=[chatbot, chat_citations, chat_input]
                )

    login_btn.click(
        login_fn,
        inputs=[username, password],
        outputs=[
            login_box,
            app_panel,
            login_err,
            login_err,
            hybrid_llm_model,
            hybrid_reranker_model,
            vector_llm_model,
            vector_reranker_model,
            chat_llm_model,
            chat_reranker_model
        ]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866)
