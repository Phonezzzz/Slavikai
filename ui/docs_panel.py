from __future__ import annotations

from PySide6.QtWidgets import (
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
from shared.models import VectorSearchResult


class DocsPanel(QWidget):
    """Панель просмотра ProjectFacts и поиска по индексам code/docs."""

    def __init__(self) -> None:
        super().__init__()
        self.memory = MemoryManager("memory/memory.db")
        self.vectors = VectorIndex("memory/vectors.db")

        layout = QVBoxLayout()
        # Project facts
        facts_row = QHBoxLayout()
        facts_row.addWidget(QLabel("Project:"))
        self.project_input = QLineEdit()
        self.project_input.setPlaceholderText("project_id")
        self.facts_btn = QPushButton("🔍 Факты")
        self.facts_btn.clicked.connect(self.load_facts)
        facts_row.addWidget(self.project_input)
        facts_row.addWidget(self.facts_btn)
        layout.addLayout(facts_row)

        # Docs search
        docs_row = QHBoxLayout()
        self.docs_query = QLineEdit()
        self.docs_query.setPlaceholderText("Поиск по docs-индексу")
        self.docs_btn = QPushButton("🔍 Docs index")
        self.docs_btn.clicked.connect(self.search_docs)
        docs_row.addWidget(self.docs_query)
        docs_row.addWidget(self.docs_btn)
        layout.addLayout(docs_row)

        # Code search
        code_row = QHBoxLayout()
        self.code_query = QLineEdit()
        self.code_query.setPlaceholderText("Поиск по code-индексу")
        self.code_btn = QPushButton("🔍 Code index")
        self.code_btn.clicked.connect(self.search_code)
        code_row.addWidget(self.code_query)
        code_row.addWidget(self.code_btn)
        layout.addLayout(code_row)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.preview_btn = QPushButton("👁 Предпросмотр выделенного")
        self.preview_btn.clicked.connect(self.show_preview)
        layout.addWidget(self.preview_btn)
        layout.addWidget(self.output)
        self.setLayout(layout)

    def load_facts(self) -> None:
        project = self.project_input.text().strip()
        if not project:
            return
        facts = self.memory.get_project_facts(project)
        self.output.clear()
        if not facts:
            self.output.append("Факты не найдены.")
            return
        for fact in facts:
            self.output.append(
                f"[{fact.timestamp}] {fact.content}\nmeta={fact.meta} tags={','.join(fact.tags)}\n"
            )

    def search_docs(self) -> None:
        query = self.docs_query.text().strip()
        if not query:
            return
        results = self.vectors.search(query, namespace="docs", top_k=5)
        self._render_index_results(results)

    def search_code(self) -> None:
        query = self.code_query.text().strip()
        if not query:
            return
        results = self.vectors.search(query, namespace="code", top_k=5)
        self._render_index_results(results)

    def _render_index_results(self, results: list[VectorSearchResult]) -> None:
        self.output.clear()
        if not results:
            self.output.append("Совпадений нет.")
            return
        for res in results:
            meta = f" meta={res.meta}" if res.meta else ""
            query = self.docs_query.text().strip() or self.code_query.text().strip()
            snippet = self._highlight(query, res.snippet)
            self.output.append(f"{res.path} [{res.score:.3f}]{meta}\n→ {snippet}\n")

    def show_preview(self) -> None:
        cursor = self.output.textCursor()
        selected = cursor.selectedText()
        if not selected:
            QMessageBox.information(self, "Предпросмотр", "Выделите текст в результатах.")
            return
        QMessageBox.information(self, "Предпросмотр", selected)

    def _highlight(self, query: str, text: str) -> str:
        if not query:
            return text
        return text.replace(query, f"<mark>{query}</mark>")
