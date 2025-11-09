import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import time
from sentence_transformers import util

st.set_page_config(
    page_title="Inception Embedding Test Suite",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8005"

def check_service_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def get_current_settings():
    try:
        response = requests.get(f"{API_BASE_URL}/current_settings", timeout=2)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def reload_service(settings: Dict):
    try:
        response = requests.post(
            f"{API_BASE_URL}/reload_service",
            json=settings,
            timeout=120
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Service reload failed: {str(e)}")
        return None

def embed_query(text: str) -> Optional[Dict]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/embed/query",
            json={"text": text},
            timeout=30
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Query embedding failed: {str(e)}")
        return None

def embed_document(text: str) -> Optional[Dict]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/embed/text",
            data=text.encode('utf-8'),
            headers={"Content-Type": "text/plain"},
            timeout=120
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Document embedding failed: {str(e)}")
        return None

def embed_batch(documents: List[Dict]) -> Optional[List[Dict]]:
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/embed/batch",
            json={"documents": documents},
            timeout=300
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Batch embedding failed: {str(e)}")
        return None

st.title("🔮 Inception Embedding Test Suite")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    health = check_service_health()
    current_model = None
    server_settings = get_current_settings()
    
    if health:
        status_color = "green" if health.get("status") == "healthy" else "orange"
        st.markdown(f"**Status:** :{status_color}[{health.get('status', 'unknown').upper()}]")
        st.markdown(f"**Model Loaded:** {health.get('model_loaded', False)}")
        
        if health.get('model_info'):
            model_info = health['model_info']
            current_model = model_info.get('name', 'Unknown')
            st.markdown(f"**Current Model:** `{current_model}`")
            if model_info.get('supports_prompts'):
                available_prompts = model_info.get('available_prompts', [])
                st.markdown(f"**Available Prompts:** {', '.join(available_prompts)}")
        
        st.markdown(f"**GPU Available:** {health.get('gpu_available', False)}")
        
        if health.get('prompt_config'):
            prompt_config = health['prompt_config']
            st.markdown("**Prompt Config:**")
            if prompt_config.get('use_query_prompt'):
                st.markdown(f"  - Query: `{prompt_config.get('query_prompt_name')}`")
            if prompt_config.get('use_document_prompt'):
                st.markdown(f"  - Document: `{prompt_config.get('document_prompt_name')}`")
    else:
        st.markdown("**Status:** :red[OFFLINE]")
    
    st.divider()
    
    st.subheader("🤖 Model Selection")
    
    available_models = [
        "freelawproject/modernbert-embed-base_finetune_512",
        "Qwen/Qwen3-Embedding-0.6B"
    ]
    
    current_index = 0
    if current_model:
        try:
            current_index = available_models.index(current_model)
        except ValueError:
            current_index = 0
    
    selected_model = st.selectbox(
        "Embedding Model",
        available_models,
        index=current_index,
        key="model_selector"
    )
    
    st.divider()
    
    st.subheader("Model Parameters")
    
    chunk_size = st.number_input(
        "Chunk Size (characters)",
        min_value=100,
        max_value=100000,
        value=server_settings.get('chunk_size', 2048) if server_settings else 2048,
        step=128,
        key="chunk_size"
    )
    
    chunk_overlap = st.number_input(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=10000,
        value=server_settings.get('chunk_overlap', 200) if server_settings else 200,
        step=50,
        key="chunk_overlap"
    )
    
    st.subheader("Text Constraints")
    
    min_text_length = st.number_input(
        "Min Text Length",
        min_value=1,
        max_value=1000,
        value=server_settings.get('min_text_length', 1) if server_settings else 1,
        step=1,
        key="min_text_length"
    )
    
    max_query_length = st.number_input(
        "Max Query Length",
        min_value=100,
        max_value=10000,
        value=server_settings.get('max_query_length', 1000) if server_settings else 1000,
        step=100,
        key="max_query_length"
    )
    
    max_text_length = st.number_input(
        "Max Text Length",
        min_value=1000,
        max_value=100000000,
        value=server_settings.get('max_text_length', 10000000) if server_settings else 10000000,
        step=100000,
        key="max_text_length"
    )
    
    st.subheader("Processing Settings")
    
    max_batch_size = st.number_input(
        "Max Batch Size",
        min_value=1,
        max_value=1000,
        value=server_settings.get('max_batch_size', 100) if server_settings else 100,
        step=10,
        key="max_batch_size"
    )
    
    processing_batch_size = st.number_input(
        "Processing Batch Size",
        min_value=1,
        max_value=64,
        value=server_settings.get('processing_batch_size', 8) if server_settings else 8,
        step=1,
        key="processing_batch_size"
    )
    
    max_workers = st.number_input(
        "Max Workers",
        min_value=1,
        max_value=32,
        value=server_settings.get('max_workers', 4) if server_settings else 4,
        step=1,
        key="max_workers"
    )
    
    st.subheader("System Settings")
    
    force_cpu = st.checkbox(
        "Force CPU",
        value=server_settings.get('force_cpu', False) if server_settings else False,
        key="force_cpu"
    )
    
    enable_metrics = st.checkbox(
        "Enable Metrics",
        value=server_settings.get('enable_metrics', True) if server_settings else True,
        key="enable_metrics"
    )
    
    st.divider()
    
    new_config = {
        "transformer_model_name": selected_model,
        "transformer_model_version": "main",
        "use_query_prompt": True,
        "query_prompt_name": "query",
        "use_document_prompt": False,
        "document_prompt_name": None,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "min_text_length": min_text_length,
        "max_query_length": max_query_length,
        "max_text_length": max_text_length,
        "max_batch_size": max_batch_size,
        "processing_batch_size": processing_batch_size,
        "max_workers": max_workers,
        "force_cpu": force_cpu,
        "enable_metrics": enable_metrics
    }
    
    settings_changed = False
    if server_settings:
        for key in new_config:
            if key in server_settings and new_config[key] != server_settings[key]:
                settings_changed = True
                break
    
    if settings_changed:
        st.warning("⚠️ Settings have been modified")
        if st.button("🔄 Apply Changes & Reload Service", type="primary", use_container_width=True):
            with st.spinner("Reloading service with new settings..."):
                result = reload_service(new_config)
                if result and result.get('status') == 'success':
                    st.success(f"✅ {result.get('message')}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ Failed to reload service")
    
    if st.button("📋 Copy Config as JSON", use_container_width=True):
        st.code(json.dumps(new_config, indent=2))

tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Embedding", "🔍 Query & Search", "📦 Batch Processing", "📊 Analytics"])

with tab1:
    st.header("Document Embedding Generator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        doc_source = st.radio(
            "Document Source",
            ["Paste Text", "Upload File", "Sample Documents"],
            horizontal=True,
        )
    
    document_text = ""
    
    if doc_source == "Paste Text":
        document_text = st.text_area(
            "Document Text",
            height=400,
            placeholder="Paste your document text here...",
            key="doc_paste",
        )
    
    elif doc_source == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=["txt", "md"],
        )
        if uploaded_file:
            document_text = uploaded_file.read().decode("utf-8")
            st.text_area(
                "Uploaded Content Preview",
                value=document_text[:1000] + ("..." if len(document_text) > 1000 else ""),
                height=200,
                disabled=True,
            )
    
    elif doc_source == "Sample Documents":
        samples = {
            "Legal Opinion (Short)": "The Supreme Court held that the First Amendment protects freedom of speech and expression. This fundamental right is essential to democracy and cannot be abridged without compelling governmental interest. The Court emphasized that prior restraint is particularly disfavored and must meet strict scrutiny standards.",
            "Legal Opinion (Medium)": """In this landmark case, the Court examined the scope of copyright protection for software interfaces. The defendant argued that functional elements of an API are not copyrightable, while the plaintiff maintained that creative expression in code structure deserves protection.

The Court analyzed the fair use doctrine, considering the purpose and character of use, the nature of the copyrighted work, the amount used, and market effect. After careful consideration of precedent and public policy, the Court concluded that using API declarations to create a compatible implementation constitutes fair use.

This decision has significant implications for software interoperability and innovation. The Court recognized that allowing copyright to prevent interface compatibility would harm competition and technological progress. However, the ruling carefully limits its holding to the specific facts presented.""",
            "Legal Opinion (Long)": """The case before this Court involves a complex intersection of intellectual property law, technological innovation, and public policy considerations. The plaintiff, a major software corporation, alleges that the defendant's use of certain application programming interface (API) declarations constitutes copyright infringement.

Background: The defendant created a new software platform intended to be compatible with the plaintiff's existing ecosystem. To achieve this compatibility, the defendant reimplemented certain API specifications, using the same method names, parameters, and organizational structure, but wrote entirely new underlying code.

The plaintiff argues that the defendant's use of the API structure constitutes copying of creative expression protected by copyright. The defendant counters with three main arguments: first, that functional elements are not copyrightable; second, that even if copyrightable, the use qualifies as fair use; and third, that copyright protection in this context would harm innovation and competition.

Analysis: The Court must first determine whether the API declarations at issue are copyrightable subject matter. We acknowledge that software can contain both functional and creative elements. The question is whether the specific organizational structure and naming conventions used in these APIs cross the threshold of creative expression.

Previous decisions have established that copyright protects expression but not ideas, processes, or systems. The Ninth Circuit's decision in Oracle v. Google provides instructive guidance, though we must carefully consider the specific circumstances presented here.

After reviewing the record, we find that while individual method names might lack sufficient creativity, the overall selection, arrangement, and organizational structure of the API reflects creative choices that could merit copyright protection. However, this finding does not end our inquiry.

Fair Use Analysis: Even assuming copyright protection, we must evaluate whether defendant's use constitutes fair use under 17 U.S.C. § 107. This requires consideration of four statutory factors.

First Factor - Purpose and Character: The defendant's use was transformative, creating a new platform while maintaining compatibility. This weighs in favor of fair use. The commercial nature of the use weighs against fair use, but does not preclude it.

Second Factor - Nature of Work: The API declarations are functional in nature, which weighs toward fair use. Creative works receive stronger protection than functional works.

Third Factor - Amount Used: The defendant used only what was necessary to achieve compatibility. While this included substantial portions of the API, the copying was limited to declarations, not implementation.

Fourth Factor - Market Effect: This is perhaps the most significant factor. The plaintiff argues that the defendant's platform competes directly in the marketplace. However, the evidence shows that the defendant's platform expanded the market rather than merely substituting for plaintiff's offering.

Conclusion: Balancing all factors, we conclude that the defendant's use constitutes fair use. Allowing copyright to prevent interface compatibility would create monopolistic control over software platforms and harm innovation. This holding is limited to cases involving reimplementation for compatibility purposes where only functional elements are copied.

The judgment is reversed and remanded for proceedings consistent with this opinion."""
        }
        
        selected_sample = st.selectbox(
            "Select Sample Document",
            list(samples.keys()),
        )
        document_text = samples[selected_sample]
        st.text_area(
            "Sample Content",
            value=document_text,
            height=300,
            disabled=True,
        )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Generate Embeddings", type="primary", use_container_width=True):
            if document_text:
                with st.spinner("Processing document..."):
                    start_time = time.time()
                    result = embed_document(document_text)
                    elapsed = time.time() - start_time
                    
                    if result:
                        st.session_state.doc_embeddings = result
                        st.session_state.doc_text = document_text
                        st.session_state.doc_process_time = elapsed
                        st.success(f"✅ Generated {len(result['embeddings'])} chunks in {elapsed:.2f}s")
                    else:
                        st.error("Failed to generate embeddings")
            else:
                st.warning("Please provide document text")
    
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            if 'doc_embeddings' in st.session_state:
                del st.session_state.doc_embeddings
            if 'doc_text' in st.session_state:
                del st.session_state.doc_text
            st.rerun()
    
    if 'doc_embeddings' in st.session_state:
        st.divider()
        
        embeddings = st.session_state.doc_embeddings['embeddings']
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Total Chunks", len(embeddings))
        with metric_col2:
            avg_chunk_len = sum(len(e['chunk']) for e in embeddings) / len(embeddings)
            st.metric("Avg Chunk Length", f"{avg_chunk_len:.0f} chars")
        with metric_col3:
            st.metric("Embedding Dims", len(embeddings[0]['embedding']))
        with metric_col4:
            st.metric("Process Time", f"{st.session_state.doc_process_time:.2f}s")
        
        st.subheader("📑 Document Chunks")
        
        chunk_display = st.selectbox(
            "Display Mode",
            ["Expandable Chunks", "Full Text", "Statistics Only"],
        )
        
        if chunk_display == "Expandable Chunks":
            for i, chunk_data in enumerate(embeddings, 1):
                with st.expander(f"**Chunk {i}** ({len(chunk_data['chunk'])} characters)", expanded=(i==1)):
                    st.markdown(chunk_data['chunk'])
                    
                    show_embedding = st.checkbox(f"Show embedding vector", key=f"show_emb_{i}")
                    if show_embedding:
                        st.caption("First 20 dimensions:")
                        st.bar_chart(chunk_data['embedding'][:20])
                        with st.expander("Full embedding vector"):
                            st.json(chunk_data['embedding'])
        
        elif chunk_display == "Full Text":
            for i, chunk_data in enumerate(embeddings, 1):
                st.markdown(f"**Chunk {i}:**")
                st.markdown(chunk_data['chunk'])
                st.divider()
        
        elif chunk_display == "Statistics Only":
            stats_data = []
            for i, chunk_data in enumerate(embeddings, 1):
                emb_array = np.array(chunk_data['embedding'])
                stats_data.append({
                    "Chunk": i,
                    "Characters": len(chunk_data['chunk']),
                    "Words": len(chunk_data['chunk'].split()),
                    "Embedding Mean": f"{emb_array.mean():.4f}",
                    "Embedding Std": f"{emb_array.std():.4f}",
                    "Embedding Norm": f"{np.linalg.norm(emb_array):.4f}"
                })
            st.dataframe(stats_data, use_container_width=True)

with tab2:
    st.header("Query Embedding & Document Search")
    
    query_text = st.text_area(
        "Search Query",
        height=100,
        placeholder="Enter your search query here...",
        key="query_input",
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Embed Query", type="primary", use_container_width=True):
            if query_text:
                with st.spinner("Generating query embedding..."):
                    start_time = time.time()
                    result = embed_query(query_text)
                    elapsed = time.time() - start_time
                    
                    if result:
                        st.session_state.query_embedding = result['embedding']
                        st.session_state.query_text = query_text
                        st.session_state.query_time = elapsed
                        st.success(f"✅ Query embedded in {elapsed:.2f}s")
                    else:
                        st.error("Failed to generate query embedding")
            else:
                st.warning("Please enter a query")
    
    if 'query_embedding' in st.session_state:
        st.divider()
        
        qemb = st.session_state.query_embedding
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dimensions", len(qemb))
        with col2:
            st.metric("Vector Norm", f"{np.linalg.norm(qemb):.4f}")
        with col3:
            st.metric("Generation Time", f"{st.session_state.query_time:.3f}s")
        
        with st.expander("📊 Query Embedding Visualization"):
            st.caption("First 50 dimensions:")
            st.line_chart(qemb[:50])
        
        if 'doc_embeddings' in st.session_state:
            st.divider()
            st.subheader("🎯 Similarity Search Results")
            
            doc_embeddings = st.session_state.doc_embeddings['embeddings']
            
            doc_embedding_vectors = [chunk_data['embedding'] for chunk_data in doc_embeddings]
            
            similarities_tensor = util.cos_sim(qemb, doc_embedding_vectors)
            similarities_scores = similarities_tensor[0].cpu().numpy()
            
            similarities = []
            for idx, chunk_data in enumerate(doc_embeddings):
                similarities.append({
                    'chunk_number': chunk_data['chunk_number'],
                    'similarity': float(similarities_scores[idx]),
                    'chunk': chunk_data['chunk']
                })
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            num_results = st.slider(
                "Number of results to show",
                1,
                len(similarities),
                min(5, len(similarities)),
            )
            
            st.markdown(f"**Top {num_results} Most Similar Chunks:**")
            
            for i, result in enumerate(similarities[:num_results], 1):
                similarity_pct = result['similarity'] * 100
                
                if similarity_pct >= 80:
                    color = "🟢"
                elif similarity_pct >= 60:
                    color = "🟡"
                else:
                    color = "🔴"
                
                with st.expander(
                    f"{color} **Rank {i}** - Chunk {result['chunk_number']} - Similarity: {similarity_pct:.2f}%",
                    expanded=(i==1)
                ):
                    st.progress(result['similarity'])
                    st.markdown(result['chunk'])
            
            st.divider()
            
            all_sims = [s['similarity'] for s in similarities]
            st.subheader("📈 Similarity Distribution")
            st.bar_chart(all_sims)
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Max Similarity", f"{max(all_sims)*100:.2f}%")
            with stats_col2:
                st.metric("Mean Similarity", f"{np.mean(all_sims)*100:.2f}%")
            with stats_col3:
                st.metric("Min Similarity", f"{min(all_sims)*100:.2f}%")
            with stats_col4:
                st.metric("Std Dev", f"{np.std(all_sims)*100:.2f}%")
        
        else:
            st.info("💡 Generate document embeddings in the 'Document Embedding' tab to search against them.")

with tab3:
    st.header("Batch Processing")
    
    num_documents = st.number_input(
        "Number of Documents",
        min_value=1,
        max_value=server_settings.get('max_batch_size', 100) if server_settings else 100,
        value=3,
        step=1,
    )
    
    batch_documents = []
    
    for i in range(num_documents):
        with st.expander(f"📄 Document {i+1}", expanded=(i==0)):
            doc_id = st.number_input(
                f"Document ID",
                min_value=0,
                value=i+1,
                key=f"batch_id_{i}",
            )
            doc_text = st.text_area(
                f"Text",
                height=150,
                placeholder=f"Enter text for document {i+1}...",
                key=f"batch_text_{i}",
            )
            batch_documents.append({"id": doc_id, "text": doc_text})
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🚀 Process Batch", type="primary", use_container_width=True):
            valid_docs = [doc for doc in batch_documents if doc["text"].strip()]
            
            if valid_docs:
                with st.spinner(f"Processing {len(valid_docs)} documents..."):
                    start_time = time.time()
                    results = embed_batch(valid_docs)
                    elapsed = time.time() - start_time
                    
                    if results:
                        st.session_state.batch_results = results
                        st.session_state.batch_time = elapsed
                        st.success(f"✅ Processed {len(results)} documents in {elapsed:.2f}s")
                    else:
                        st.error("Batch processing failed")
            else:
                st.warning("Please enter text for at least one document")
    
    if 'batch_results' in st.session_state:
        st.divider()
        
        results = st.session_state.batch_results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents Processed", len(results))
        with col2:
            total_chunks = sum(len(doc['embeddings']) for doc in results)
            st.metric("Total Chunks", total_chunks)
        with col3:
            st.metric("Processing Time", f"{st.session_state.batch_time:.2f}s")
        
        st.subheader("📊 Batch Summary")
        
        summary_data = []
        for doc in results:
            summary_data.append({
                "Doc ID": doc['id'],
                "Chunks": len(doc['embeddings']),
                "Total Chars": sum(len(e['chunk']) for e in doc['embeddings']),
                "Avg Chunk Size": int(sum(len(e['chunk']) for e in doc['embeddings']) / len(doc['embeddings']))
            })
        
        st.dataframe(summary_data, use_container_width=True)
        
        st.subheader("📑 Detailed Results")
        
        selected_doc = st.selectbox(
            "Select Document to View",
            [f"Document {doc['id']}" for doc in results],
        )
        
        doc_index = int(selected_doc.split()[-1]) - 1
        if doc_index < len(results):
            doc = results[doc_index]
            
            for chunk in doc['embeddings']:
                with st.expander(f"Chunk {chunk['chunk_number']} ({len(chunk['chunk'])} chars)"):
                    st.markdown(chunk['chunk'])
                    
                    if st.checkbox(f"Show embedding", key=f"batch_emb_{doc['id']}_{chunk['chunk_number']}"):
                        st.caption("First 20 dimensions:")
                        st.bar_chart(chunk['embedding'][:20])

with tab4:
    st.header("Analytics & Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Service Metrics")
        if enable_metrics:
            try:
                metrics_response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
                if metrics_response.status_code == 200:
                    st.text_area(
                        "Prometheus Metrics",
                        value=metrics_response.text,
                        height=400,
                    )
                else:
                    st.warning("Metrics endpoint returned an error")
            except:
                st.error("Could not fetch metrics from service")
        else:
            st.info("Metrics are disabled in the configuration")
    
    with col2:
        st.subheader("📊 Session Statistics")
        
        stats = {}
        
        if 'doc_embeddings' in st.session_state:
            stats['Documents Embedded'] = 1
            stats['Document Chunks Generated'] = len(st.session_state.doc_embeddings['embeddings'])
            stats['Document Processing Time'] = f"{st.session_state.doc_process_time:.2f}s"
        
        if 'query_embedding' in st.session_state:
            stats['Queries Embedded'] = 1
            stats['Query Processing Time'] = f"{st.session_state.query_time:.3f}s"
        
        if 'batch_results' in st.session_state:
            stats['Batch Jobs Run'] = 1
            stats['Batch Documents'] = len(st.session_state.batch_results)
            stats['Batch Processing Time'] = f"{st.session_state.batch_time:.2f}s"
        
        if stats:
            for key, value in stats.items():
                st.metric(key, value)
        else:
            st.info("No operations performed yet in this session")
        
        st.divider()
        
        st.subheader("🔧 Configuration Summary")
        if server_settings:
            config_df = pd.DataFrame([{k: str(v) for k, v in server_settings.items()}]).T
            config_df.columns = ['Value']
            st.dataframe(config_df, use_container_width=True)
        
        if st.button("🔄 Reset Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.caption(f"🌐 API Base URL: `{API_BASE_URL}`")
with footer_col2:
    st.caption("🔮 Inception Embedding Service")
with footer_col3:
    if health:
        st.caption(f"✅ Service Online")
    else:
        st.caption(f"❌ Service Offline")