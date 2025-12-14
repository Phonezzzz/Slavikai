from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from memory.memory_manager import MemoryManager
from memory.vector_index import VectorIndex
from shared.models import MemoryKind


class MemoryView(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.manager = MemoryManager("memory/memory.db")
        self.vectors = VectorIndex("memory/vectors.db")
        layout = QVBoxLayout()
        controls = QHBoxLayout()
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Поиск по памяти / индексу")
        self.kind_select = QComboBox()
        self.kind_select.addItems(["all", "note", "user_pref", "project_fact"])
        self.namespace_select = QComboBox()
        self.namespace_select.addItems(["code", "docs"])
        self.search_mem_btn = QPushButton("🔍 Память")
        self.search_mem_btn.clicked.connect(self.search_memory)
        self.search_idx_btn = QPushButton("🔍 Индекс")
        self.search_idx_btn.clicked.connect(self.search_index)
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("project_id для фактов")
        self.search_facts_btn = QPushButton("🔍 Факты")
        self.search_facts_btn.clicked.connect(self.search_project_facts)
        self.sort_select = QComboBox()
        self.sort_select.addItems(["relevance", "recency"])
        self.preview_btn = QPushButton("👁 Предпросмотр выделенного")
        self.preview_btn.clicked.connect(self.show_preview)
        controls.addWidget(self.query_input)
        controls.addWidget(self.kind_select)
        controls.addWidget(self.namespace_select)
        controls.addWidget(self.search_mem_btn)
        controls.addWidget(self.search_idx_btn)
        controls.addWidget(self.project_input)
        controls.addWidget(self.search_facts_btn)
        controls.addWidget(QLabel("Сортировка:"))
        controls.addWidget(self.sort_select)
        controls.addWidget(self.preview_btn)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.refresh_btn = QPushButton("🔄 Обновить память")
        self.refresh_btn.clicked.connect(self.refresh)
        layout.addLayout(controls)
        layout.addWidget(self.text)
        layout.addWidget(self.refresh_btn)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        items = self.manager.get_recent(10)
        self.text.clear()
        for item in items:
            self.text.append(f"[{item.timestamp}] {item.content[:150]}...\n")

    def search_memory(self) -> None:
        query = self.query_input.text().strip()
        kind_val = self.kind_select.currentText()
        kind = None
        if kind_val != "all":
            kind = MemoryKind(kind_val)
        items = (
            self.manager.search(query, kind=kind) if query else self.manager.get_recent(20, kind)
        )
        self.text.clear()
        for item in items:
            meta = item.meta or {}
            snippet = self._highlight(self.query_input.text().strip(), item.content[:400])
            self.text.append(
                f"[{item.timestamp}] {item.kind.value}: {snippet}\n"
                f"tags={','.join(item.tags)} meta={meta}\n"
            )

    def search_index(self) -> None:
        query = self.query_input.text().strip()
        if not query:
            return
        namespace = self.namespace_select.currentText()
        results = self.vectors.search(query, namespace=namespace, top_k=8)
        sort_mode = self.sort_select.currentText()
        if sort_mode == "recency":
            # для простоты — оставляем исходный порядок, предполагаем что индекс свежий
            pass
        else:
            results = sorted(results, key=lambda r: r.score, reverse=True)
        self.text.clear()
        for res in results:
            snippet = self._highlight(query, res.snippet)
            self.text.append(f"{res.path} [{res.score:.3f}]\n→ {snippet}\n")

    def search_project_facts(self) -> None:
        project = self.project_input.text().strip()
        if not project:
            return
        facts = self.manager.get_project_facts(project)
        self.text.clear()
        for fact in facts:
            snippet = self._highlight(self.query_input.text().strip(), fact.content)
            self.text.append(f"[{fact.timestamp}] project={project}\n{snippet}\nmeta={fact.meta}\n")

    def show_preview(self) -> None:
        cursor = self.text.textCursor()
        selected = cursor.selectedText()
        if not selected:
            QMessageBox.information(self, "Предпросмотр", "Выделите текст в списке результатов.")
            return
        QMessageBox.information(self, "Предпросмотр", selected)

    def _highlight(self, query: str, text: str) -> str:
        if not query:
            return text
        return text.replace(query, f"<mark>{query}</mark>")
