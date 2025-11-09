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

from constants import SUPPORTED_EXTENSIONS

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


class FileCollectionWorker(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, paths, use_subdirs=False, is_directory=False):
        super().__init__()
        self.paths = paths
        self.use_subdirs = use_subdirs
        self.is_directory = is_directory
        self._is_running = True

    def run(self):
        try:
            all_files = []

            if self.is_directory:
                directory = Path(self.paths[0])

                if self.use_subdirs:
                    for ext in SUPPORTED_EXTENSIONS:
                        pattern = f"**/*{ext}"
                        for filepath in directory.glob(pattern):
                            if not self._is_running:
                                return
                            if filepath.is_file():
                                all_files.append(str(filepath))
                else:
                    for ext in SUPPORTED_EXTENSIONS:
                        pattern = f"*{ext}"
                        for filepath in directory.glob(pattern):
                            if not self._is_running:
                                return
                            if filepath.is_file():
                                all_files.append(str(filepath))
            else:
                for filepath in self.paths:
                    if not self._is_running:
                        return
                    path = Path(filepath)
                    if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file():
                        all_files.append(str(filepath))

            if self._is_running:
                self.finished.emit(all_files)

        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))

    def stop(self):
        self._is_running = False


class FileProcessWorker(QThread):
    progress = Signal(int, int)
    batch_processed = Signal(list)
    finished = Signal()
    error = Signal(str)

    def __init__(self, filepaths, batch_size=500):
        super().__init__()
        self.filepaths = filepaths
        self.batch_size = batch_size
        self._is_running = True

    def run(self):
        try:
            total_files = len(self.filepaths)

            for batch_start in range(0, total_files, self.batch_size):
                if not self._is_running:
                    break

                batch_end = min(batch_start + self.batch_size, total_files)
                batch_paths = self.filepaths[batch_start:batch_end]

                files = []
                for filepath in batch_paths:
                    files.append(
                        ('files', (Path(filepath).name, open(filepath, 'rb'), 'application/octet-stream'))
                    )

                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/extract_files_batch",
                        files=files,
                        timeout=300
                    )

                    if response.status_code == 200:
                        result = response.json()
                        batch_results = []
                        for file_data in result['files']:
                            batch_results.append({
                                'global_index': batch_start + file_data['id'],
                                'path': batch_paths[file_data['id']],
                                'filename': file_data['filename'],
                                'text': file_data['text'],
                                'success': file_data['success'],
                                'error': file_data.get('error'),
                                'size': file_data['size']
                            })

                        if self._is_running:
                            self.batch_processed.emit(batch_results)
                            self.progress.emit(batch_end, total_files)
                    else:
                        raise Exception(f"Server returned status {response.status_code}")
                finally:
                    for _, (_, file_obj, _) in files:
                        file_obj.close()

            if self._is_running:
                self.finished.emit()
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))

    def stop(self):
        self._is_running = False


class ChunkWorker(QThread):
    progress = Signal(int, int)
    batch_chunked = Signal(list)
    finished = Signal()
    error = Signal(str)

    def __init__(self, documents, batch_size=100):
        super().__init__()
        self.documents = documents
        self.batch_size = batch_size
        self._is_running = True

    def run(self):
        try:
            doc_list = list(self.documents.items())
            total_docs = len(doc_list)
            
            for batch_start in range(0, total_docs, self.batch_size):
                if not self._is_running:
                    break

                batch_end = min(batch_start + self.batch_size, total_docs)
                batch_docs = []

                for idx, (doc_id, text) in enumerate(doc_list[batch_start:batch_end]):
                    batch_docs.append({
                        "id": doc_id,
                        "text": text
                    })

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/chunk_texts_batch",
                    json={"documents": batch_docs},
                    timeout=300
                )

                if response.status_code == 200:
                    result = response.json()
                    if self._is_running:
                        self.batch_chunked.emit(result['results'])
                        self.progress.emit(batch_end, total_docs)
                else:
                    raise Exception(f"Server returned status {response.status_code}")

            if self._is_running:
                self.finished.emit()
        except Exception as e:
            if self._is_running:
                self.error.emit(str(e))

    def stop(self):
        self._is_running = False


class EmbedWorker(QThread):
    progress = Signal(int, int)
    batch_embedded = Signal(list)
    finished = Signal()
    error = Signal(str)

    def __init__(self, chunked_documents, batch_size=100):
        super().__init__()
        self.chunked_documents = chunked_documents
        self.batch_size = batch_size
        self._is_running = True

    def run(self):
        try:
            doc_list = list(self.chunked_documents.items())
            total_docs = len(doc_list)

            for batch_start in range(0, total_docs, self.batch_size):
                if not self._is_running:
                    break

                batch_end = min(batch_start + self.batch_size, total_docs)
                batch_docs = []

                for doc_id, chunks in doc_list[batch_start:batch_end]:
                    combined_text = "\n\n".join(chunks)
                    batch_docs.append({
                        "id": doc_id,
                        "text": combined_text
                    })

                response = requests.post(
                    f"{API_BASE_URL}/api/v1/embed/batch",
                    json={"documents": batch_docs},
                    timeout=600
                )

                if response.status_code == 200:
                    result = response.json()
                    if self._is_running:
                        self.batch_embedded.emit(result)
                        self.progress.emit(batch_end, total_docs)
                else:
                    raise Exception(f"Server returned status {response.status_code}")

            if self._is_running:
                self.finished.emit()
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
        self.batch_files = {}
        self.extracted_texts = {}
        self.chunked_texts = {}
        self.all_embeddings = []

        self.active_workers = []
        self.file_collection_worker = None
        self.file_process_worker = None
        self.chunk_worker = None
        self.embed_worker = None

        self.setup_ui()
        self.setup_timers()
        self.check_health()
        self.fetch_settings()

    def closeEvent(self, event):
        self.health_timer.stop()

        if self.file_collection_worker:
            self.file_collection_worker.stop()
            self.file_collection_worker.quit()
            self.file_collection_worker.wait(1000)

        if self.file_process_worker:
            self.file_process_worker.stop()
            self.file_process_worker.quit()
            self.file_process_worker.wait(1000)

        if self.chunk_worker:
            self.chunk_worker.stop()
            self.chunk_worker.quit()
            self.chunk_worker.wait(1000)

        if self.embed_worker:
            self.embed_worker.stop()
            self.embed_worker.quit()
            self.embed_worker.wait(1000)

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

        self.text_cleaning_combo = QComboBox()
        self.text_cleaning_combo.addItems([
            "ASCII Only",
            "ASCII + Extended Latin",
            "Unicode Safe",
            "Whitespace Only",
            "No Cleaning"
        ])
        self.text_cleaning_combo.currentTextChanged.connect(self.settings_changed)
        processing_layout.addWidget(QLabel("Text Cleaning Mode:"))
        processing_layout.addWidget(self.text_cleaning_combo)

        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)

        system_group = QGroupBox("System Settings")
        system_layout = QVBoxLayout()

        self.force_cpu_check = QCheckBox("Force CPU")
        self.force_cpu_check.stateChanged.connect(self.settings_changed)
        system_layout.addWidget(self.force_cpu_check)

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
        tabs.addTab(self.create_query_tab(), "Query and Search")
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
        self.similarity_table.setColumnCount(4)
        self.similarity_table.setHorizontalHeaderLabels(["Rank", "Document", "Chunk #", "Similarity"])
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

        selection_group = QGroupBox("File Selection")
        selection_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.select_files_btn = QPushButton("Select Files...")
        self.select_files_btn.clicked.connect(self.browse_batch_files)
        self.select_dir_btn = QPushButton("Select Directory...")
        self.select_dir_btn.clicked.connect(self.browse_batch_directory)
        btn_layout.addWidget(self.select_files_btn)
        btn_layout.addWidget(self.select_dir_btn)
        btn_layout.addStretch()
        selection_layout.addLayout(btn_layout)

        options_layout = QHBoxLayout()
        self.subdirs_check = QCheckBox("Include subdirectories when selecting directory")
        self.subdirs_check.setChecked(False)
        options_layout.addWidget(self.subdirs_check)
        options_layout.addStretch()
        selection_layout.addLayout(options_layout)

        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)

        self.collection_progress = QProgressBar()
        self.collection_progress.setVisible(False)
        self.collection_progress.setTextVisible(True)
        self.collection_progress.setFormat("Collecting files: %v")
        layout.addWidget(self.collection_progress)

        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout()

        self.extract_progress = QProgressBar()
        self.extract_progress.setVisible(False)
        process_layout.addWidget(QLabel("Text Extraction:"))
        process_layout.addWidget(self.extract_progress)

        self.chunk_progress = QProgressBar()
        self.chunk_progress.setVisible(False)
        process_layout.addWidget(QLabel("Text Chunking:"))
        process_layout.addWidget(self.chunk_progress)

        self.embed_progress = QProgressBar()
        self.embed_progress.setVisible(False)
        process_layout.addWidget(QLabel("Embedding Generation:"))
        process_layout.addWidget(self.embed_progress)

        btn_row = QHBoxLayout()
        self.process_files_btn = QPushButton("Process Files")
        self.process_files_btn.clicked.connect(self.process_files)
        self.process_files_btn.setEnabled(False)
        self.chunk_text_btn = QPushButton("Chunk Text")
        self.chunk_text_btn.clicked.connect(self.chunk_text)
        self.chunk_text_btn.setEnabled(False)
        self.process_batch_btn = QPushButton("Process Batch")
        self.process_batch_btn.clicked.connect(self.process_batch)
        self.process_batch_btn.setEnabled(False)
        btn_row.addWidget(self.process_files_btn)
        btn_row.addWidget(self.chunk_text_btn)
        btn_row.addWidget(self.process_batch_btn)
        btn_row.addStretch()
        process_layout.addLayout(btn_row)

        process_group.setLayout(process_layout)
        layout.addWidget(process_group)

        splitter = QSplitter(Qt.Vertical)

        files_group = QGroupBox("Files")
        files_layout = QVBoxLayout()

        select_btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all_files)
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all_files)
        select_btn_layout.addWidget(self.select_all_btn)
        select_btn_layout.addWidget(self.deselect_all_btn)
        select_btn_layout.addStretch()
        files_layout.addLayout(select_btn_layout)

        self.batch_files_table = QTableWidget()
        self.batch_files_table.setColumnCount(4)
        self.batch_files_table.setHorizontalHeaderLabels(["Include", "Status", "File Name", "Size"])
        self.batch_files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.batch_files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.batch_files_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.batch_files_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.batch_files_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.batch_files_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.batch_files_table.setSelectionMode(QTableWidget.SingleSelection)
        self.batch_files_table.itemSelectionChanged.connect(self.display_selected_file_text)
        files_layout.addWidget(self.batch_files_table)
        files_group.setLayout(files_layout)
        splitter.addWidget(files_group)

        text_group = QGroupBox("File Content")
        text_layout = QVBoxLayout()
        self.batch_file_display = QTextEdit()
        self.batch_file_display.setReadOnly(True)
        self.batch_file_display.setPlaceholderText("Select a file to view its content...")
        text_layout.addWidget(self.batch_file_display)
        text_group.setLayout(text_layout)
        splitter.addWidget(text_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

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
        self.batch_summary_table.setHorizontalHeaderLabels(["File Name", "Chunks", "Total Chars", "Avg Chunk Size"])
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

        cleaning_mode = settings.get('text_cleaning_mode', 'ascii_only')
        cleaning_map = {
            'ascii_only': 0,
            'ascii_extended': 1,
            'unicode_safe': 2,
            'whitespace_only': 3,
            'no_cleaning': 4
        }
        cleaning_index = cleaning_map.get(cleaning_mode, 0)
        self.text_cleaning_combo.setCurrentIndex(cleaning_index)
        
        self.apply_btn.setEnabled(False)

    def settings_changed(self):
        self.apply_btn.setEnabled(True)
        self.apply_btn.setText("⚠ Apply Changes & Reload Service")

    def get_new_config(self):
        cleaning_modes = [
            "ascii_only",
            "ascii_extended",
            "unicode_safe",
            "whitespace_only",
            "no_cleaning"
        ]

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
            "text_cleaning_mode": cleaning_modes[self.text_cleaning_combo.currentIndex()]
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
            "Text Files (*.txt *.md *.html);;All Files (*)"
        )
        if filename:
            try:
                with open(filename, 'rb') as f:
                    content = f.read()
                
                files = [('files', (Path(filename).name, content, 'application/octet-stream'))]
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/extract_files_batch",
                    files=files,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result['files'][0]['success']:
                        self.doc_text_edit.setText(result['files'][0]['text'])
                    else:
                        QMessageBox.critical(self, "Error", f"Failed to extract text: {result['files'][0]['error']}")
                else:
                    QMessageBox.critical(self, "Error", f"Server error: {response.status_code}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read file: {str(e)}")

    def browse_batch_files(self):
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files",
            "",
            "Text Files (*.txt *.md *.html);;All Files (*)"
        )
        if filenames:
            self.start_file_collection(filenames, is_directory=False)

    def browse_batch_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory"
        )
        if directory:
            self.start_file_collection([directory], is_directory=True)

    def start_file_collection(self, paths, is_directory=False):
        if self.file_collection_worker and self.file_collection_worker.isRunning():
            self.file_collection_worker.stop()
            self.file_collection_worker.wait()

        self.batch_files = {}
        self.extracted_texts = {}
        self.chunked_texts = {}
        self.all_embeddings = []
        self.batch_files_table.setRowCount(0)
        self.batch_file_display.clear()

        self.collection_progress.setVisible(True)
        self.collection_progress.setMaximum(0)
        self.collection_progress.setFormat("Collecting files...")

        self.select_files_btn.setEnabled(False)
        self.select_dir_btn.setEnabled(False)
        self.process_files_btn.setEnabled(False)

        use_subdirs = self.subdirs_check.isChecked() if is_directory else False

        self.file_collection_worker = FileCollectionWorker(paths, use_subdirs, is_directory)
        self.file_collection_worker.finished.connect(self.on_collection_complete)
        self.file_collection_worker.error.connect(self.on_collection_error)
        self.file_collection_worker.start()

    def on_collection_complete(self, all_files):
        self.collection_progress.setFormat("Updating table...")
        self.statusBar().showMessage(f"Updating table with {len(all_files)} files...")

        self.batch_files_table.setUpdatesEnabled(False)
        self.batch_files_table.setRowCount(len(all_files))

        for idx, filepath in enumerate(all_files):
            path = Path(filepath)

            file_info = {
                'path': filepath,
                'text': '',
                'success': False,
                'error': None,
                'size': 0
            }
            self.batch_files[idx] = file_info

            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.batch_files_table.setCellWidget(idx, 0, checkbox_widget)

            status_item = QTableWidgetItem("⋯")
            status_item.setForeground(QColor("gray"))
            self.batch_files_table.setItem(idx, 1, status_item)

            self.batch_files_table.setItem(idx, 2, QTableWidgetItem(path.name))
            self.batch_files_table.setItem(idx, 3, QTableWidgetItem("-"))

        self.batch_files_table.setUpdatesEnabled(True)

        self.collection_progress.setVisible(False)
        self.select_files_btn.setEnabled(True)
        self.select_dir_btn.setEnabled(True)
        self.process_files_btn.setEnabled(len(self.batch_files) > 0)
        self.statusBar().showMessage(f"Collected {len(self.batch_files)} files", 3000)

    def on_collection_error(self, error):
        self.collection_progress.setVisible(False)
        self.select_files_btn.setEnabled(True)
        self.select_dir_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Error collecting files: {error}")

    def process_files(self):
        selected_files = []
        
        for row in range(self.batch_files_table.rowCount()):
            checkbox_widget = self.batch_files_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    file_info = self.batch_files[row]
                    selected_files.append(file_info['path'])

        if not selected_files:
            QMessageBox.warning(self, "Warning", "No files selected")
            return

        self.process_files_btn.setEnabled(False)
        self.process_files_btn.setText("Processing...")
        self.extract_progress.setVisible(True)
        self.extract_progress.setMaximum(len(selected_files))
        self.extract_progress.setValue(0)

        self.file_process_worker = FileProcessWorker(selected_files, batch_size=500)
        self.file_process_worker.progress.connect(self.on_extract_progress)
        self.file_process_worker.batch_processed.connect(self.on_batch_extracted)
        self.file_process_worker.finished.connect(self.on_extract_complete)
        self.file_process_worker.error.connect(self.on_extract_error)
        self.file_process_worker.start()

    def on_extract_progress(self, current, total):
        self.extract_progress.setValue(current)
        self.statusBar().showMessage(f"Extracting text: {current}/{total}")

    def on_batch_extracted(self, batch):
        for file_info in batch:
            idx = file_info['global_index']
            self.batch_files[idx]['text'] = file_info['text']
            self.batch_files[idx]['success'] = file_info['success']
            self.batch_files[idx]['error'] = file_info['error']
            self.batch_files[idx]['size'] = file_info['size']

            if file_info['success']:
                self.extracted_texts[idx] = file_info['text']

            status_item = QTableWidgetItem("✓" if file_info['success'] else "✗")
            status_item.setForeground(QColor("green") if file_info['success'] else QColor("red"))
            self.batch_files_table.setItem(idx, 1, status_item)
            self.batch_files_table.setItem(idx, 3, QTableWidgetItem(str(file_info['size'])))

        QApplication.processEvents()

    def on_extract_complete(self):
        self.extract_progress.setVisible(False)
        self.process_files_btn.setEnabled(True)
        self.process_files_btn.setText("Process Files")
        self.chunk_text_btn.setEnabled(len(self.extracted_texts) > 0)
        self.statusBar().showMessage(f"Extracted {len(self.extracted_texts)} files", 3000)

    def on_extract_error(self, error):
        self.extract_progress.setVisible(False)
        self.process_files_btn.setEnabled(True)
        self.process_files_btn.setText("Process Files")
        QMessageBox.critical(self, "Error", f"Extraction failed: {error}")

    def chunk_text(self):
        if not self.extracted_texts:
            QMessageBox.warning(self, "Warning", "No extracted text to chunk")
            return

        self.chunk_text_btn.setEnabled(False)
        self.chunk_text_btn.setText("Chunking...")
        self.chunk_progress.setVisible(True)
        self.chunk_progress.setMaximum(len(self.extracted_texts))
        self.chunk_progress.setValue(0)

        self.chunk_worker = ChunkWorker(self.extracted_texts, batch_size=100)
        self.chunk_worker.progress.connect(self.on_chunk_progress)
        self.chunk_worker.batch_chunked.connect(self.on_batch_chunked)
        self.chunk_worker.finished.connect(self.on_chunk_complete)
        self.chunk_worker.error.connect(self.on_chunk_error)
        self.chunk_worker.start()

    def on_chunk_progress(self, current, total):
        self.chunk_progress.setValue(current)
        self.statusBar().showMessage(f"Chunking text: {current}/{total}")

    def on_batch_chunked(self, batch):
        for result in batch:
            if result['success']:
                self.chunked_texts[result['id']] = result['chunks']
        QApplication.processEvents()

    def on_chunk_complete(self):
        self.chunk_progress.setVisible(False)
        self.chunk_text_btn.setEnabled(True)
        self.chunk_text_btn.setText("Chunk Text")
        self.process_batch_btn.setEnabled(len(self.chunked_texts) > 0)
        self.statusBar().showMessage(f"Chunked {len(self.chunked_texts)} documents", 3000)

    def on_chunk_error(self, error):
        self.chunk_progress.setVisible(False)
        self.chunk_text_btn.setEnabled(True)
        self.chunk_text_btn.setText("Chunk Text")
        QMessageBox.critical(self, "Error", f"Chunking failed: {error}")

    def process_batch(self):
        if not self.chunked_texts:
            QMessageBox.warning(self, "Warning", "No chunked text to process")
            return

        self.process_batch_btn.setEnabled(False)
        self.process_batch_btn.setText("Generating Embeddings...")
        self.embed_progress.setVisible(True)
        self.embed_progress.setMaximum(len(self.chunked_texts))
        self.embed_progress.setValue(0)

        start_time = time.time()
        self.batch_start_time = start_time

        self.embed_worker = EmbedWorker(self.chunked_texts, batch_size=100)
        self.embed_worker.progress.connect(self.on_embed_progress)
        self.embed_worker.batch_embedded.connect(self.on_batch_embedded)
        self.embed_worker.finished.connect(self.on_embed_complete)
        self.embed_worker.error.connect(self.on_embed_error)
        self.embed_worker.start()

    def on_embed_progress(self, current, total):
        self.embed_progress.setValue(current)
        self.statusBar().showMessage(f"Generating embeddings: {current}/{total}")

    def on_batch_embedded(self, batch):
        self.all_embeddings.extend(batch)
        QApplication.processEvents()

    def on_embed_complete(self):
        elapsed = time.time() - self.batch_start_time
        
        self.embed_progress.setVisible(False)
        self.process_batch_btn.setEnabled(True)
        self.process_batch_btn.setText("Process Batch")

        self.batch_results = self.all_embeddings
        self.batch_time = elapsed

        total_chunks = sum(len(doc['embeddings']) for doc in self.all_embeddings)

        self.batch_docs_label.setText(f"Documents: {len(self.all_embeddings)}")
        self.batch_chunks_label.setText(f"Chunks: {total_chunks}")
        self.batch_time_label.setText(f"Time: {elapsed:.2f}s")

        self.batch_summary_table.setRowCount(len(self.all_embeddings))
        for i, doc in enumerate(self.all_embeddings):
            doc_id = doc['id']
            filename = Path(self.batch_files[doc_id]['path']).name
            self.batch_summary_table.setItem(i, 0, QTableWidgetItem(filename))
            self.batch_summary_table.setItem(i, 1, QTableWidgetItem(str(len(doc['embeddings']))))
            total_chars = sum(len(e['chunk']) for e in doc['embeddings'])
            self.batch_summary_table.setItem(i, 2, QTableWidgetItem(str(total_chars)))
            avg_size = int(total_chars / len(doc['embeddings'])) if doc['embeddings'] else 0
            self.batch_summary_table.setItem(i, 3, QTableWidgetItem(str(avg_size)))

        self.batch_results_group.setVisible(True)
        self.statusBar().showMessage(f"Generated embeddings for {len(self.all_embeddings)} documents in {elapsed:.2f}s", 5000)

    def on_embed_error(self, error):
        self.embed_progress.setVisible(False)
        self.process_batch_btn.setEnabled(True)
        self.process_batch_btn.setText("Process Batch")
        QMessageBox.critical(self, "Error", f"Embedding generation failed: {error}")

    def select_all_files(self):
        for row in range(self.batch_files_table.rowCount()):
            checkbox_widget = self.batch_files_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isEnabled():
                    checkbox.setChecked(True)

    def deselect_all_files(self):
        for row in range(self.batch_files_table.rowCount()):
            checkbox_widget = self.batch_files_table.cellWidget(row, 0)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isEnabled():
                    checkbox.setChecked(False)

    def display_selected_file_text(self):
        selected_rows = self.batch_files_table.selectedIndexes()
        if not selected_rows:
            self.batch_file_display.clear()
            return

        row = selected_rows[0].row()
        if row < len(self.batch_files):
            file_info = self.batch_files[row]
            if file_info['success'] and file_info['text']:
                self.batch_file_display.setPlainText(file_info['text'])
            elif file_info['error']:
                self.batch_file_display.setPlainText(f"Error: {file_info['error']}")
            else:
                self.batch_file_display.setPlainText("No text extracted yet")

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

        if self.batch_results:
            self.compute_similarities_batch()
        elif self.doc_embeddings:
            self.compute_similarities_single()

        self.embed_query_btn.setEnabled(True)
        self.embed_query_btn.setText("Embed Query")
        self.statusBar().showMessage(f"Query embedded in {self.query_time:.3f}s", 5000)

    def query_embedding_failed(self, error):
        QMessageBox.critical(self, "Error", f"Failed to embed query: {error}")
        self.embed_query_btn.setEnabled(True)
        self.embed_query_btn.setText("Embed Query")

    def compute_similarities_single(self):
        doc_embeddings = self.doc_embeddings['embeddings']

        qemb = np.array(self.query_embedding)
        doc_vecs = [np.array(chunk['embedding']) for chunk in doc_embeddings]

        similarities = []
        for idx, doc_vec in enumerate(doc_vecs):
            sim = np.dot(qemb, doc_vec) / (np.linalg.norm(qemb) * np.linalg.norm(doc_vec))
            similarities.append({
                'document': 'Single Document',
                'chunk_number': doc_embeddings[idx]['chunk_number'],
                'similarity': float(sim),
                'chunk': doc_embeddings[idx]['chunk']
            })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        num_results = min(6, len(similarities))
        self.similarity_results = similarities[:num_results]

        self.similarity_table.setRowCount(num_results)
        for i, result in enumerate(self.similarity_results):
            self.similarity_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.similarity_table.setItem(i, 1, QTableWidgetItem(result['document']))
            self.similarity_table.setItem(i, 2, QTableWidgetItem(str(result['chunk_number'])))
            self.similarity_table.setItem(i, 3, QTableWidgetItem(f"{result['similarity']*100:.2f}%"))

        self.chunk_display.clear()
        self.similarity_group.setVisible(True)

    def compute_similarities_batch(self):
        qemb = np.array(self.query_embedding)
        similarities = []

        for doc_result in self.batch_results:
            doc_id = doc_result['id']
            doc_name = Path(self.batch_files[doc_id]['path']).name
            
            for chunk_data in doc_result['embeddings']:
                chunk_vec = np.array(chunk_data['embedding'])
                sim = np.dot(qemb, chunk_vec) / (np.linalg.norm(qemb) * np.linalg.norm(chunk_vec))
                similarities.append({
                    'document': doc_name,
                    'chunk_number': chunk_data['chunk_number'],
                    'similarity': float(sim),
                    'chunk': chunk_data['chunk']
                })

        similarities.sort(key=lambda x: x['similarity'], reverse=True)

        num_results = min(10, len(similarities))
        self.similarity_results = similarities[:num_results]

        self.similarity_table.setRowCount(num_results)
        for i, result in enumerate(self.similarity_results):
            self.similarity_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.similarity_table.setItem(i, 1, QTableWidgetItem(result['document']))
            self.similarity_table.setItem(i, 2, QTableWidgetItem(str(result['chunk_number'])))
            self.similarity_table.setItem(i, 3, QTableWidgetItem(f"{result['similarity']*100:.2f}%"))

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


def main():
    app = QApplication(sys.argv)

    app.setStyle('Fusion')

    window = InceptionGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()