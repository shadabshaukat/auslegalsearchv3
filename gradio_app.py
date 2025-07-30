"""
AUSLegalSearchv3 Gradio UI:
- Tabs for Hybrid Search, Vector Search, RAG (with supersystem prompt), and Conversational Chat with Top K, source cards, and modern chat interface.
"""

import gradio as gr
import requests
import os
import html
import json

API_ROOT = os.environ.get("AUSLEGALSEARCH_API_URL", "http://localhost:8000")
SESS = type("Session", (), {"user": None, "auth": None})()
SESS.auth = None

DEFAULT_SYSTEM_PROMPT = """You are an expert Australian legal research and compliance AI assistant.
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
        return [
            (
                m.get("display_name", m.get("model_name", m.get("id", "Unknown"))),
                json.dumps(m)
            )
            for m in all_models
            if m.get("id") or m.get("ocid") or m.get("model_id")
        ] or [("No OCI GenAI models found. Check Oracle Console/config.", "")]
    except Exception as exc:
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

def format_context_cards(sources, chunk_metadata, context_chunks):
    if not sources and not context_chunks:
        return ""
    # Render a stylish source card for each
    cards = []
    for idx in range(max(len(sources), len(context_chunks))):
        meta = (chunk_metadata[idx] if idx < len(chunk_metadata) else {}) or {}
        citation = meta.get("citation") or meta.get("url") or (sources[idx] if idx < len(sources) else "?")
        url = meta.get("url")
        url_part = f'<a href="{url}" target="_blank">{html.escape(str(citation))}</a>' if url else html.escape(str(citation))
        text = context_chunks[idx] if idx < len(context_chunks) else ""
        excerpt = html.escape(text[:600]) + ("..." if len(text) > 600 else "")
        card = f"""
        <div class='chat-source-card'>
            <div class='chat-source-cite'>{url_part}</div>
            <div class='chat-source-excerpt'>{excerpt}</div>
        </div>
        """
        cards.append(card)
    return "<div class='chat-source-cards-ct'>" + "".join(cards) + "</div>"

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
    return format_context_cards([h.get("citation","?") for h in hits], [h.get("chunk_metadata") or {} for h in hits], [h.get("text","") for h in hits])

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
    return "<ul>" + "".join(f"<li>{html.escape(h.get('text','')[:150])}{'...' if len(h.get('text',''))>150 else ''}</li>" for h in hits) + "</ul>"

def rag_chatbot(question, llm_source, ollama_model, oci_model_info_json, rag_top_k, system_prompt, temperature, top_p, max_tokens, repeat_penalty):
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
    params = {
        "context_chunks": context_chunks or [],
        "sources": sources or [],
        "chunk_metadata": chunk_metadata or [],
        "custom_prompt": system_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty
    }
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
            **params,
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
            **params,
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
    context_html = format_context_cards(sources, chunk_metadata, context_chunks)
    return answer_html, context_html, sources

def conversational_chat_fn(message, llm_source, ollama_model, oci_model_info_json, top_k, history, system_prompt, temperature, top_p, max_tokens, repeat_penalty):
    chat_history = history or []
    req = {
        "llm_source": "ollama" if llm_source == "Local Ollama" else "oci_genai",
        "model": ollama_model,
        "message": message,
        "chat_history": chat_history,
        "system_prompt": system_prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
        "top_k": top_k,
        "oci_config": {},
    }
    if llm_source == "OCI GenAI":
        try:
            model_info = json.loads(oci_model_info_json) if oci_model_info_json else {}
        except Exception:
            model_info = {}
        req["oci_config"] = {
            "compartment_id": os.environ.get("OCI_COMPARTMENT_OCID", ""),
            "model_id": model_info.get("id") or model_info.get("ocid") or model_info.get("model_id") or os.environ.get("OCI_GENAI_MODEL_OCID", ""),
            "region": os.environ.get("OCI_REGION", "ap-sydney-1"),
        }
    try:
        resp = requests.post(f"{API_ROOT}/chat/conversation", json=req, auth=SESS.auth, timeout=120)
        data = resp.json()
        answer = data.get("answer", "") if isinstance(data, dict) else str(data)
        sources = data.get("sources", [])
        chunk_metadata = data.get("chunk_metadata", [])
        context_chunks = data.get("context_chunks", [])

        # ADD: Extract proper text for OCI GenAI if answer is a full JSON (chat_response)
        def extract_oci_text(ans):
            # Handle dict
            if isinstance(ans, dict) and "chat_response" in ans:
                try:
                    choices = ans["chat_response"].get("choices", [])
                    if choices:
                        content_list = choices[0]["message"].get("content", [])
                        if content_list and isinstance(content_list[0], dict):
                            return content_list[0].get("text", "")
                except Exception:
                    return str(ans)
            # Sometimes backend may json.dumps() the answer:
            if isinstance(ans, str):
                try:
                    asdict = json.loads(ans)
                    return extract_oci_text(asdict)
                except Exception:
                    return ans
            return str(ans)

        # If answer looks like OCI GenAI "Assistant: { ... }", extract inner
        if isinstance(answer, dict) or (isinstance(answer, str) and answer.strip().startswith("{")):
            try:
                answer_text = extract_oci_text(answer)
            except Exception:
                answer_text = str(answer)
        else:
            answer_text = answer

        formatted_answer = html.escape(answer_text).replace("\\n", "<br>").replace("\n", "<br>")
    except Exception as e:
        formatted_answer = f"Error querying chatbot API: {e}"
        sources = []
        chunk_metadata = []
        context_chunks = []
    if not isinstance(chat_history, list):
        chat_history = []
    # Append user and assistant+cards to history, with separate cards entry
    chat_history.append({"role": "user", "content": message})
    chat_history.append({
        "role": "assistant",
        "content": formatted_answer,
        "cards": format_context_cards(sources, chunk_metadata, context_chunks)
    })
    # Render as modern chat bubbles with context cards in a separate grid below answer
    def render_history(hist):
        html_out = "<div class='chatbox-ct'>"
        idx = 0
        while idx < len(hist):
            msg = hist[idx]
            if msg["role"] == "user":
                html_out += f"<div class='bubble user-bubble'><b>User:</b> {msg['content']}</div>"
            elif msg["role"] == "assistant":
                # Always show just the answer in the bubble, and the context cards right below it as grid
                html_out += (
                    "<div class='bubble assistant-bubble'><b>Assistant:</b> "
                    f"{msg['content']}</div>"
                )
                if msg.get("cards"):
                    html_out += f"<div>{msg['cards']}</div>"
            idx += 1
        html_out += "</div>"
        return html_out or "<i>No conversation yet.</i>"
    return render_history(chat_history), chat_history

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
.chatbox-ct {
    display: flex;
    flex-direction: column;
    gap: 1.2em;
    max-width: 700px;
    margin: 0 auto;
}
.bubble {
    border-radius: 12px;
    padding: 16px 20px;
    max-width: 95%;
    margin: 2px 0;
    font-size: 1.08em;
    box-shadow: 0 2.5px 8px #e7f4e9;
    transition: border .15s;
}
.user-bubble {
    background: #f0f5fd;
    align-self: flex-start;
    border-bottom-left-radius: 0;
}
.assistant-bubble {
    background: #eafbe5;
    align-self: flex-end;
    border-bottom-right-radius: 0;
}
.chat-source-cards-ct {
    display: flex;
    flex-wrap: wrap;
    gap: .7em;
    margin-top: 12px;
}
.chat-source-card {
    border: 1.7px solid #ebeff3;
    background: #fff;
    border-radius: 7px;
    padding: 7.5px 13px 9.5px 13px;
    margin-right: 0;
    min-width: 215px;
    max-width: 315px;
    min-height: 68px;
    font-size: .98em;
    display: flex;
    flex-direction: column;
    gap: 6px;
}
.chat-source-cite {
    color: #276188;
    font-weight: bold;
    margin-bottom: 2px;
}
.chat-source-excerpt {
    color: #18311a;
    font-family: Menlo, Monaco, "SFMono-Regular", monospace;
    background: #f8fafd;
    padding: 2.8px 5px 2.8px 5px;
    border-radius: 4px;
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
                system_prompt = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM_PROMPT, lines=3)
                temperature = gr.Slider(label="Temperature", value=0.1, minimum=0.0, maximum=1.5, step=0.01)
                top_p = gr.Slider(label="Top P", value=0.9, minimum=0.0, maximum=1.0, step=0.01)
                max_tokens = gr.Number(label="Max Tokens", value=1024, precision=0)
                repeat_penalty = gr.Slider(label="Repeat Penalty", value=1.1, minimum=0.5, maximum=2.0, step=0.01)
                def update_model_dropdowns(src):
                    if src == "OCI GenAI":
                        return (
                            gr.update(visible=False),
                            gr.update(choices=fetch_oci_models(), visible=True)
                        )
                    else:
                        return (
                            gr.update(choices=fetch_ollama_models(), visible=True),
                            gr.update(choices=[], visible=False)
                        )
                llm_source.change(update_model_dropdowns, inputs=[llm_source], outputs=[ollama_model, oci_model])
                question = gr.Textbox(label="Enter your legal or compliance question", lines=2)
                ask_btn = gr.Button("Ask")
                answer = gr.HTML(label="Answer", elem_id="llm-answer-box", value="")
                context = gr.HTML(label="Context / Sources", value="", show_label=False)
                ask_btn.click(
                    rag_chatbot,
                    inputs=[question, llm_source, ollama_model, oci_model, rag_top_k, system_prompt, temperature, top_p, max_tokens, repeat_penalty],
                    outputs=[answer, context, gr.State()]
                )
            with gr.Tab("Conversational Chat"):
                gr.Markdown("#### Conversational Chatbot (RAG-style: each turn uses Top K hybrid search for context, sources shown as cards)")
                chat_llm_source = gr.Dropdown(label="LLM Source", choices=["Local Ollama", "OCI GenAI"], value="Local Ollama")
                chat_ollama_model = gr.Dropdown(label="Ollama Model", choices=[], visible=True)
                chat_oci_model = gr.Dropdown(label="OCI GenAI Model", choices=[], visible=False)
                chat_top_k = gr.Number(label="Top K Context Chunks", value=10, precision=0)
                chat_system_prompt = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM_PROMPT, lines=3)
                chat_temperature = gr.Slider(label="Temperature", value=0.1, minimum=0.0, maximum=1.5, step=0.01)
                chat_top_p = gr.Slider(label="Top P", value=0.9, minimum=0.0, maximum=1.0, step=0.01)
                chat_max_tokens = gr.Number(label="Max Tokens", value=1024, precision=0)
                chat_repeat_penalty = gr.Slider(label="Repeat Penalty", value=1.1, minimum=0.5, maximum=2.0, step=0.01)
                chat_history = gr.State([])
                def update_chat_model_dropdowns(src):
                    if src == "OCI GenAI":
                        return (
                            gr.update(visible=False),
                            gr.update(choices=fetch_oci_models(), visible=True)
                        )
                    else:
                        return (
                            gr.update(choices=fetch_ollama_models(), visible=True),
                            gr.update(choices=[], visible=False)
                        )
                chat_llm_source.change(update_chat_model_dropdowns, inputs=[chat_llm_source], outputs=[chat_ollama_model, chat_oci_model])
                chat_message = gr.Textbox(label="Your message", lines=2)
                send_btn = gr.Button("Send")
                conversation_html = gr.HTML(label="Conversation", value="", show_label=False)

                # Show progress while backend call is in flight
                def show_in_progress(*_):
                    return (
                        "<div class='chatbox-ct'><div class='bubble user-bubble'>Sending message...</div><div class='bubble assistant-bubble' style='color:#aaa;'><span class='spinner'></span> Fetching response...</div></div>",
                        gr.update()
                    )

                send_btn.click(
                    show_in_progress,
                    inputs=[chat_message, chat_llm_source, chat_ollama_model, chat_oci_model,
                            chat_top_k, chat_history, chat_system_prompt, chat_temperature, chat_top_p, chat_max_tokens, chat_repeat_penalty],
                    outputs=[conversation_html, chat_history],
                    queue=False
                ).then(
                    conversational_chat_fn,
                    inputs=[chat_message, chat_llm_source, chat_ollama_model, chat_oci_model,
                            chat_top_k, chat_history, chat_system_prompt, chat_temperature, chat_top_p, chat_max_tokens, chat_repeat_penalty],
                    outputs=[conversation_html, chat_history]
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
