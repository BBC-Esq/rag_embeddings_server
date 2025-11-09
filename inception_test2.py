import sys
import json
import time
import numpy as np
import requests
from pathlib import Path
from typing import Optional, Dict, List
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QComboBox, QSpinBox, QCheckBox,
    QTabWidget, QGroupBox, QTableWidget, QTableWidgetItem, QProgressBar,
    QFileDialog, QSplitter, QScrollArea, QMessageBox, QDoubleSpinBox,
    QHeaderView, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont, QColor, QPalette


API_BASE_URL = "http://localhost:8005"


class ApiWorker(QThread):
    finished = Signal(object)
    error = Signal(str)
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._is_running = True
    
    def run(self):
        try:
            if self._is_running:
                result = self.func(*self.args, **self.kwargs)
                if self._is_running:
                    self.finished.emit(result)
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))
    
    def stop(self):
        self._is_running = False


class InceptionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Inception Embedding Test Suite")
        self.setGeometry(100, 100, 1600, 900)
        
        self.health_status = None
        self.current_settings = None
        self.doc_embeddings = None
        self.doc_text = None
        self.doc_process_time = None
        self.query_embedding = None
        self.query_text = None
        self.query_time = None
        self.batch_results = None
        self.batch_time = None
        
        self.active_workers = []
        
        self.setup_ui()
        self.setup_timers()
        self.check_health()
        self.fetch_settings()
    
    def closeEvent(self, event):
        self.health_timer.stop()
        
        for worker in self.active_workers:
            worker.stop()
            worker.quit()
            worker.wait(1000)
        
        event.accept()
    
    def add_worker(self, worker):
        self.active_workers.append(worker)
        worker.finished.connect(lambda: self.remove_worker(worker))
        worker.error.connect(lambda: self.remove_worker(worker))
    
    def remove_worker(self, worker):
        if worker in self.active_workers:
            self.active_workers.remove(worker)
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        left_panel = self.create_sidebar()
        splitter.addWidget(left_panel)
        
        right_panel = self.create_main_content()
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        
        self.statusBar().showMessage("Ready")
    
    def create_sidebar(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        
        container = QWidget()
        layout = QVBoxLayout(container)
        
        status_group = QGroupBox("Service Status")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Checking...")
        self.model_label = QLabel("Model: Unknown")
        self.gpu_label = QLabel("GPU: Unknown")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.model_label)
        status_layout.addWidget(self.gpu_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "freelawproject/modernbert-embed-base_finetune_512",
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-4B"
        ])
        self.model_combo.currentTextChanged.connect(self.settings_changed)
        model_layout.addWidget(QLabel("Embedding Model:"))
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        params_group = QGroupBox("Model Parameters")
        params_layout = QVBoxLayout()
        
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(100, 100000)
        self.chunk_size_spin.setValue(2048)
        self.chunk_size_spin.setSingleStep(128)
        self.chunk_size_spin.valueChanged.connect(self.settings_changed)
        params_layout.addWidget(QLabel("Chunk Size (characters):"))
        params_layout.addWidget(self.chunk_size_spin)
        
        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setRange(0, 10000)
        self.chunk_overlap_spin.setValue(200)
        self.chunk_overlap_spin.setSingleStep(50)
        self.chunk_overlap_spin.valueChanged.connect(self.settings_changed)
        params_layout.addWidget(QLabel("Chunk Overlap (characters):"))
        params_layout.addWidget(self.chunk_overlap_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        constraints_group = QGroupBox("Text Constraints")
        constraints_layout = QVBoxLayout()
        
        self.min_text_spin = QSpinBox()
        self.min_text_spin.setRange(1, 1000)
        self.min_text_spin.setValue(1)
        self.min_text_spin.valueChanged.connect(self.settings_changed)
        constraints_layout.addWidget(QLabel("Min Text Length:"))
        constraints_layout.addWidget(self.min_text_spin)
        
        self.max_query_spin = QSpinBox()
        self.max_query_spin.setRange(100, 10000)
        self.max_query_spin.setValue(1000)
        self.max_query_spin.setSingleStep(100)
        self.max_query_spin.valueChanged.connect(self.settings_changed)
        constraints_layout.addWidget(QLabel("Max Query Length:"))
        constraints_layout.addWidget(self.max_query_spin)
        
        self.max_text_spin = QSpinBox()
        self.max_text_spin.setRange(1000, 100000000)
        self.max_text_spin.setValue(10000000)
        self.max_text_spin.setSingleStep(100000)
        self.max_text_spin.valueChanged.connect(self.settings_changed)
        constraints_layout.addWidget(QLabel("Max Text Length:"))
        constraints_layout.addWidget(self.max_text_spin)
        
        constraints_group.setLayout(constraints_layout)
        layout.addWidget(constraints_group)
        
        processing_group = QGroupBox("Processing Settings")
        processing_layout = QVBoxLayout()
        
        self.max_batch_spin = QSpinBox()
        self.max_batch_spin.setRange(1, 1000)
        self.max_batch_spin.setValue(100)
        self.max_batch_spin.setSingleStep(10)
        self.max_batch_spin.valueChanged.connect(self.settings_changed)
        processing_layout.addWidget(QLabel("Max Batch Size:"))
        processing_layout.addWidget(self.max_batch_spin)
        
        self.processing_batch_spin = QSpinBox()
        self.processing_batch_spin.setRange(1, 64)
        self.processing_batch_spin.setValue(8)
        self.processing_batch_spin.valueChanged.connect(self.settings_changed)
        processing_layout.addWidget(QLabel("Processing Batch Size:"))
        processing_layout.addWidget(self.processing_batch_spin)
        
        self.max_workers_spin = QSpinBox()
        self.max_workers_spin.setRange(1, 32)
        self.max_workers_spin.setValue(4)
        self.max_workers_spin.valueChanged.connect(self.settings_changed)
        processing_layout.addWidget(QLabel("Max Workers:"))
        processing_layout.addWidget(self.max_workers_spin)
        
        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)
        
        system_group = QGroupBox("System Settings")
        system_layout = QVBoxLayout()
        
        self.force_cpu_check = QCheckBox("Force CPU")
        self.force_cpu_check.stateChanged.connect(self.settings_changed)
        system_layout.addWidget(self.force_cpu_check)
        
        self.enable_metrics_check = QCheckBox("Enable Metrics")
        self.enable_metrics_check.setChecked(True)
        self.enable_metrics_check.stateChanged.connect(self.settings_changed)
        system_layout.addWidget(self.enable_metrics_check)
        
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)
        
        self.apply_btn = QPushButton("Apply Changes & Reload Service")
        self.apply_btn.clicked.connect(self.reload_service)
        self.apply_btn.setEnabled(False)
        layout.addWidget(self.apply_btn)
        
        self.copy_config_btn = QPushButton("Copy Config as JSON")
        self.copy_config_btn.clicked.connect(self.copy_config)
        layout.addWidget(self.copy_config_btn)
        
        layout.addStretch()
        
        scroll.setWidget(container)
        return scroll
    
    def create_main_content(self):
        tabs = QTabWidget()
        
        tabs.addTab(self.create_document_tab(), "Document Embedding")
        tabs.addTab(self.create_query_tab(), "Query & Search")
        tabs.addTab(self.create_batch_tab(), "Batch Processing")
        
        return tabs
    
    def create_document_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Document Source:"))
        self.doc_source_group = QButtonGroup()
        self.paste_radio = QRadioButton("Paste Text")
        self.upload_radio = QRadioButton("Upload File")
        self.paste_radio.setChecked(True)
        self.doc_source_group.addButton(self.paste_radio)
        self.doc_source_group.addButton(self.upload_radio)
        source_layout.addWidget(self.paste_radio)
        source_layout.addWidget(self.upload_radio)
        source_layout.addStretch()
        layout.addLayout(source_layout)
        
        self.doc_source_group.buttonClicked.connect(self.doc_source_changed)
        
        self.doc_text_edit = QTextEdit()
        self.doc_text_edit.setPlaceholderText("Paste your document text here...")
        layout.addWidget(self.doc_text_edit)
        
        self.upload_btn = QPushButton("Browse File...")
        self.upload_btn.clicked.connect(self.browse_file)
        self.upload_btn.setVisible(False)
        layout.addWidget(self.upload_btn)
        
        btn_layout = QHBoxLayout()
        self.gen_doc_btn = QPushButton("Generate Embeddings")
        self.gen_doc_btn.clicked.connect(self.generate_document_embeddings)
        self.clear_doc_btn = QPushButton("Clear Results")
        self.clear_doc_btn.clicked.connect(self.clear_document_results)
        btn_layout.addWidget(self.gen_doc_btn)
        btn_layout.addWidget(self.clear_doc_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        self.doc_results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        metrics_layout = QHBoxLayout()
        self.doc_chunks_label = QLabel("Chunks: -")
        self.doc_avg_label = QLabel("Avg Length: -")
        self.doc_dims_label = QLabel("Dims: -")
        self.doc_time_label = QLabel("Time: -")
        metrics_layout.addWidget(self.doc_chunks_label)
        metrics_layout.addWidget(self.doc_avg_label)
        metrics_layout.addWidget(self.doc_dims_label)
        metrics_layout.addWidget(self.doc_time_label)
        results_layout.addLayout(metrics_layout)
        
        self.doc_chunks_table = QTableWidget()
        self.doc_chunks_table.setColumnCount(3)
        self.doc_chunks_table.setHorizontalHeaderLabels(["Chunk #", "Length", "Content Preview"])
        self.doc_chunks_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.doc_chunks_table)
        
        self.doc_results_group.setLayout(results_layout)
        self.doc_results_group.setVisible(False)
        layout.addWidget(self.doc_results_group)
        
        return widget
    
    def create_query_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("Search Query:"))
        self.query_text_edit = QTextEdit()
        self.query_text_edit.setPlaceholderText("Enter your search query here...")
        self.query_text_edit.setMaximumHeight(100)
        self.query_text_edit.setMinimumHeight(100)
        layout.addWidget(self.query_text_edit)
        
        self.embed_query_btn = QPushButton("Embed Query")
        self.embed_query_btn.clicked.connect(self.generate_query_embedding)
        layout.addWidget(self.embed_query_btn)
        
        self.similarity_group = QGroupBox("Similarity Search Results")
        similarity_layout = QVBoxLayout()
        
        self.similarity_table = QTableWidget()
        self.similarity_table.setColumnCount(3)
        self.similarity_table.setHorizontalHeaderLabels(["Rank", "Chunk #", "Similarity"])
        self.similarity_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.similarity_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.similarity_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.similarity_table.setSelectionMode(QTableWidget.SingleSelection)
        self.similarity_table.itemSelectionChanged.connect(self.display_selected_chunk)
        similarity_layout.addWidget(self.similarity_table)
        
        similarity_layout.addWidget(QLabel("Chunk Content:"))
        self.chunk_display = QTextEdit()
        self.chunk_display.setReadOnly(True)
        self.chunk_display.setPlaceholderText("Select a row to view chunk content...")
        similarity_layout.addWidget(self.chunk_display, 1)
        
        self.similarity_group.setLayout(similarity_layout)
        layout.addWidget(self.similarity_group, 1)
        
        return widget
    
    def create_batch_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Number of Documents:"))
        self.num_docs_spin = QSpinBox()
        self.num_docs_spin.setRange(1, 100)
        self.num_docs_spin.setValue(3)
        self.num_docs_spin.valueChanged.connect(self.update_batch_docs)
        num_layout.addWidget(self.num_docs_spin)
        num_layout.addStretch()
        layout.addLayout(num_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.batch_container = QWidget()
        self.batch_layout = QVBoxLayout(self.batch_container)
        scroll.setWidget(self.batch_container)
        layout.addWidget(scroll)
        
        self.batch_doc_widgets = []
        self.update_batch_docs()
        
        self.process_batch_btn = QPushButton("Process Batch")
        self.process_batch_btn.clicked.connect(self.process_batch)
        layout.addWidget(self.process_batch_btn)
        
        self.batch_results_group = QGroupBox("Batch Results")
        batch_results_layout = QVBoxLayout()
        
        batch_metrics_layout = QHBoxLayout()
        self.batch_docs_label = QLabel("Documents: -")
        self.batch_chunks_label = QLabel("Chunks: -")
        self.batch_time_label = QLabel("Time: -")
        batch_metrics_layout.addWidget(self.batch_docs_label)
        batch_metrics_layout.addWidget(self.batch_chunks_label)
        batch_metrics_layout.addWidget(self.batch_time_label)
        batch_results_layout.addLayout(batch_metrics_layout)
        
        self.batch_summary_table = QTableWidget()
        self.batch_summary_table.setColumnCount(4)
        self.batch_summary_table.setHorizontalHeaderLabels(["Doc ID", "Chunks", "Total Chars", "Avg Chunk Size"])
        self.batch_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        batch_results_layout.addWidget(self.batch_summary_table)
        
        self.batch_results_group.setLayout(batch_results_layout)
        self.batch_results_group.setVisible(False)
        layout.addWidget(self.batch_results_group)
        
        return widget
    
    def setup_timers(self):
        self.health_timer = QTimer()
        self.health_timer.timeout.connect(self.check_health)
        self.health_timer.start(10000)
    
    def check_health(self):
        worker = ApiWorker(self._check_service_health)
        worker.finished.connect(self.update_health_status)
        worker.error.connect(lambda e: self.update_health_status(None))
        self.add_worker(worker)
        worker.start()
    
    def _check_service_health(self):
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def update_health_status(self, health):
        self.health_status = health
        if health:
            status = health.get('status', 'unknown').upper()
            color = "green" if status == "HEALTHY" else "orange"
            self.status_label.setText(f"Status: {status}")
            self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            
            if health.get('model_info'):
                model_info = health['model_info']
                self.model_label.setText(f"Model: {model_info.get('name', 'Unknown')}")
            
            gpu = "Yes" if health.get('gpu_available') else "No"
            self.gpu_label.setText(f"GPU Available: {gpu}")
        else:
            self.status_label.setText("Status: OFFLINE")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.model_label.setText("Model: Unknown")
            self.gpu_label.setText("GPU: Unknown")
    
    def fetch_settings(self):
        worker = ApiWorker(self._get_current_settings)
        worker.finished.connect(self.update_settings)
        self.add_worker(worker)
        worker.start()
    
    def _get_current_settings(self):
        try:
            response = requests.get(f"{API_BASE_URL}/current_settings", timeout=2)
            return response.json() if response.status_code == 200 else None
        except:
            return None
    
    def update_settings(self, settings):
        if not settings:
            return
        
        self.current_settings = settings
        
        model_name = settings.get('transformer_model_name')
        index = self.model_combo.findText(model_name)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        
        self.chunk_size_spin.setValue(settings.get('chunk_size', 2048))
        self.chunk_overlap_spin.setValue(settings.get('chunk_overlap', 200))
        self.min_text_spin.setValue(settings.get('min_text_length', 1))
        self.max_query_spin.setValue(settings.get('max_query_length', 1000))
        self.max_text_spin.setValue(settings.get('max_text_length', 10000000))
        self.max_batch_spin.setValue(settings.get('max_batch_size', 100))
        self.processing_batch_spin.setValue(settings.get('processing_batch_size', 8))
        self.max_workers_spin.setValue(settings.get('max_workers', 4))
        self.force_cpu_check.setChecked(settings.get('force_cpu', False))
        self.enable_metrics_check.setChecked(settings.get('enable_metrics', True))
        
        self.apply_btn.setEnabled(False)
    
    def settings_changed(self):
        self.apply_btn.setEnabled(True)
        self.apply_btn.setText("⚠ Apply Changes & Reload Service")
    
    def get_new_config(self):
        return {
            "transformer_model_name": self.model_combo.currentText(),
            "transformer_model_version": "main",
            "chunk_size": self.chunk_size_spin.value(),
            "chunk_overlap": self.chunk_overlap_spin.value(),
            "min_text_length": self.min_text_spin.value(),
            "max_query_length": self.max_query_spin.value(),
            "max_text_length": self.max_text_spin.value(),
            "max_batch_size": self.max_batch_spin.value(),
            "processing_batch_size": self.processing_batch_spin.value(),
            "max_workers": self.max_workers_spin.value(),
            "force_cpu": self.force_cpu_check.isChecked(),
            "enable_metrics": self.enable_metrics_check.isChecked()
        }
    
    def reload_service(self):
        new_config = self.get_new_config()
        self.apply_btn.setEnabled(False)
        self.apply_btn.setText("Reloading...")
        
        worker = ApiWorker(self._reload_service, new_config)
        worker.finished.connect(self.reload_complete)
        worker.error.connect(self.reload_failed)
        self.add_worker(worker)
        worker.start()
    
    def _reload_service(self, settings):
        response = requests.post(
            f"{API_BASE_URL}/reload_service",
            json=settings,
            timeout=120
        )
        return response.json() if response.status_code == 200 else None
    
    def reload_complete(self, result):
        if result and result.get('status') == 'success':
            QMessageBox.information(self, "Success", result.get('message'))
            self.apply_btn.setText("Apply Changes & Reload Service")
            self.fetch_settings()
            self.check_health()
        else:
            QMessageBox.critical(self, "Error", "Failed to reload service")
            self.apply_btn.setEnabled(True)
            self.apply_btn.setText("⚠ Apply Changes & Reload Service")
    
    def reload_failed(self, error):
        QMessageBox.critical(self, "Error", f"Service reload failed: {error}")
        self.apply_btn.setEnabled(True)
        self.apply_btn.setText("⚠ Apply Changes & Reload Service")
    
    def copy_config(self):
        config = self.get_new_config()
        QApplication.clipboard().setText(json.dumps(config, indent=2))
        self.statusBar().showMessage("Configuration copied to clipboard", 3000)
    
    def doc_source_changed(self):
        if self.paste_radio.isChecked():
            self.doc_text_edit.setVisible(True)
            self.doc_text_edit.setReadOnly(False)
            self.upload_btn.setVisible(False)
        elif self.upload_radio.isChecked():
            self.doc_text_edit.setVisible(True)
            self.doc_text_edit.setReadOnly(True)
            self.upload_btn.setVisible(True)
    
    def browse_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Document",
            "",
            "Text Files (*.txt *.md);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.doc_text_edit.setText(content)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read file: {str(e)}")
    
    def generate_document_embeddings(self):
        text = self.doc_text_edit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Warning", "Please provide document text")
            return
        
        self.gen_doc_btn.setEnabled(False)
        self.gen_doc_btn.setText("Processing...")
        self.statusBar().showMessage("Generating embeddings...")
        
        worker = ApiWorker(self._embed_document, text)
        worker.finished.connect(self.document_embeddings_complete)
        worker.error.connect(self.document_embeddings_failed)
        self.add_worker(worker)
        worker.start()
    
    def _embed_document(self, text):
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/api/v1/embed/text",
            data=text.encode('utf-8'),
            headers={"Content-Type": "text/plain"},
            timeout=120
        )
        elapsed = time.time() - start_time
        if response.status_code == 200:
            result = response.json()
            result['elapsed'] = elapsed
            return result
        else:
            raise Exception(f"Server returned status {response.status_code}")
    
    def document_embeddings_complete(self, result):
        self.doc_embeddings = result
        self.doc_text = self.doc_text_edit.toPlainText()
        self.doc_process_time = result.get('elapsed', 0)
        
        embeddings = result['embeddings']
        
        self.doc_chunks_label.setText(f"Chunks: {len(embeddings)}")
        avg_len = sum(len(e['chunk']) for e in embeddings) / len(embeddings)
        self.doc_avg_label.setText(f"Avg Length: {avg_len:.0f} chars")
        self.doc_dims_label.setText(f"Dims: {len(embeddings[0]['embedding'])}")
        self.doc_time_label.setText(f"Time: {self.doc_process_time:.2f}s")
        
        self.doc_chunks_table.setRowCount(len(embeddings))
        for i, chunk_data in enumerate(embeddings):
            self.doc_chunks_table.setItem(i, 0, QTableWidgetItem(str(chunk_data['chunk_number'])))
            self.doc_chunks_table.setItem(i, 1, QTableWidgetItem(str(len(chunk_data['chunk']))))
            preview = chunk_data['chunk'][:100] + "..." if len(chunk_data['chunk']) > 100 else chunk_data['chunk']
            self.doc_chunks_table.setItem(i, 2, QTableWidgetItem(preview))
        
        self.doc_results_group.setVisible(True)
        self.gen_doc_btn.setEnabled(True)
        self.gen_doc_btn.setText("Generate Embeddings")
        self.statusBar().showMessage(f"Generated {len(embeddings)} chunks in {self.doc_process_time:.2f}s", 5000)
    
    def document_embeddings_failed(self, error):
        QMessageBox.critical(self, "Error", f"Failed to generate embeddings: {error}")
        self.gen_doc_btn.setEnabled(True)
        self.gen_doc_btn.setText("Generate Embeddings")
        self.statusBar().showMessage("Embedding generation failed", 5000)
    
    def clear_document_results(self):
        self.doc_embeddings = None
        self.doc_text = None
        self.doc_process_time = None
        self.doc_results_group.setVisible(False)
        self.doc_chunks_table.setRowCount(0)
    
    def generate_query_embedding(self):
        text = self.query_text_edit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, "Warning", "Please enter a query")
            return
        
        self.embed_query_btn.setEnabled(False)
        self.embed_query_btn.setText("Processing...")
        
        worker = ApiWorker(self._embed_query, text)
        worker.finished.connect(self.query_embedding_complete)
        worker.error.connect(self.query_embedding_failed)
        self.add_worker(worker)
        worker.start()
    
    def _embed_query(self, text):
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/api/v1/embed/query",
            json={"text": text},
            timeout=30
        )
        elapsed = time.time() - start_time
        if response.status_code == 200:
            result = response.json()
            result['elapsed'] = elapsed
            return result
        else:
            raise Exception(f"Server returned status {response.status_code}")
    
    def query_embedding_complete(self, result):
        self.query_embedding = result['embedding']
        self.query_text = self.query_text_edit.toPlainText()
        self.query_time = result.get('elapsed', 0)
        
        if self.doc_embeddings:
            self.compute_similarities()
        
        self.embed_query_btn.setEnabled(True)
        self.embed_query_btn.setText("Embed Query")
        self.statusBar().showMessage(f"Query embedded in {self.query_time:.3f}s", 5000)
    
    def query_embedding_failed(self, error):
        QMessageBox.critical(self, "Error", f"Failed to embed query: {error}")
        self.embed_query_btn.setEnabled(True)
        self.embed_query_btn.setText("Embed Query")
    
    def compute_similarities(self):
        doc_embeddings = self.doc_embeddings['embeddings']
        
        qemb = np.array(self.query_embedding)
        doc_vecs = [np.array(chunk['embedding']) for chunk in doc_embeddings]
        
        similarities = []
        for idx, doc_vec in enumerate(doc_vecs):
            sim = np.dot(qemb, doc_vec) / (np.linalg.norm(qemb) * np.linalg.norm(doc_vec))
            similarities.append({
                'chunk_number': doc_embeddings[idx]['chunk_number'],
                'similarity': float(sim),
                'chunk': doc_embeddings[idx]['chunk']
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        num_results = min(5, len(similarities))
        self.similarity_results = similarities[:num_results]
        
        self.similarity_table.setRowCount(num_results)
        for i, result in enumerate(self.similarity_results):
            self.similarity_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.similarity_table.setItem(i, 1, QTableWidgetItem(str(result['chunk_number'])))
            self.similarity_table.setItem(i, 2, QTableWidgetItem(f"{result['similarity']*100:.2f}%"))
        
        self.chunk_display.clear()
        self.similarity_group.setVisible(True)

    def display_selected_chunk(self):
        selected_rows = self.similarity_table.selectedIndexes()
        if not selected_rows:
            self.chunk_display.clear()
            return
        
        row = selected_rows[0].row()
        if hasattr(self, 'similarity_results') and row < len(self.similarity_results):
            chunk_text = self.similarity_results[row]['chunk']
            self.chunk_display.setPlainText(chunk_text)

    def update_batch_docs(self):
        while self.batch_layout.count():
            child = self.batch_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        self.batch_doc_widgets = []
        num_docs = self.num_docs_spin.value()
        
        for i in range(num_docs):
            group = QGroupBox(f"Document {i+1}")
            layout = QVBoxLayout()
            
            id_layout = QHBoxLayout()
            id_layout.addWidget(QLabel("Document ID:"))
            id_spin = QSpinBox()
            id_spin.setRange(0, 10000)
            id_spin.setValue(i + 1)
            id_layout.addWidget(id_spin)
            id_layout.addStretch()
            layout.addLayout(id_layout)
            
            text_edit = QTextEdit()
            text_edit.setPlaceholderText(f"Enter text for document {i+1}...")
            text_edit.setMaximumHeight(100)
            layout.addWidget(text_edit)
            
            group.setLayout(layout)
            self.batch_layout.addWidget(group)
            
            self.batch_doc_widgets.append({
                'group': group,
                'id_spin': id_spin,
                'text_edit': text_edit
            })
    
    def process_batch(self):
        documents = []
        for widget in self.batch_doc_widgets:
            doc_id = widget['id_spin'].value()
            text = widget['text_edit'].toPlainText()
            if text.strip():
                documents.append({"id": doc_id, "text": text})
        
        if not documents:
            QMessageBox.warning(self, "Warning", "Please enter text for at least one document")
            return
        
        self.process_batch_btn.setEnabled(False)
        self.process_batch_btn.setText("Processing...")
        
        worker = ApiWorker(self._embed_batch, documents)
        worker.finished.connect(self.batch_complete)
        worker.error.connect(self.batch_failed)
        self.add_worker(worker)
        worker.start()
    
    def _embed_batch(self, documents):
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/api/v1/embed/batch",
            json={"documents": documents},
            timeout=300
        )
        elapsed = time.time() - start_time
        if response.status_code == 200:
            result = response.json()
            return {'results': result, 'elapsed': elapsed}
        else:
            raise Exception(f"Server returned status {response.status_code}")
    
    def batch_complete(self, data):
        results = data['results']
        elapsed = data['elapsed']
        
        self.batch_results = results
        self.batch_time = elapsed
        
        total_chunks = sum(len(doc['embeddings']) for doc in results)
        
        self.batch_docs_label.setText(f"Documents: {len(results)}")
        self.batch_chunks_label.setText(f"Chunks: {total_chunks}")
        self.batch_time_label.setText(f"Time: {elapsed:.2f}s")
        
        self.batch_summary_table.setRowCount(len(results))
        for i, doc in enumerate(results):
            self.batch_summary_table.setItem(i, 0, QTableWidgetItem(str(doc['id'])))
            self.batch_summary_table.setItem(i, 1, QTableWidgetItem(str(len(doc['embeddings']))))
            total_chars = sum(len(e['chunk']) for e in doc['embeddings'])
            self.batch_summary_table.setItem(i, 2, QTableWidgetItem(str(total_chars)))
            avg_size = int(total_chars / len(doc['embeddings']))
            self.batch_summary_table.setItem(i, 3, QTableWidgetItem(str(avg_size)))
        
        self.batch_results_group.setVisible(True)
        self.process_batch_btn.setEnabled(True)
        self.process_batch_btn.setText("Process Batch")
        self.statusBar().showMessage(f"Processed {len(results)} documents in {elapsed:.2f}s", 5000)
    
    def batch_failed(self, error):
        QMessageBox.critical(self, "Error", f"Batch processing failed: {error}")
        self.process_batch_btn.setEnabled(True)
        self.process_batch_btn.setText("Process Batch")


def main():
    app = QApplication(sys.argv)
    
    app.setStyle('Fusion')
    
    window = InceptionGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()