from __future__ import annotations

import difflib
from collections.abc import Callable, Mapping
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.agent import Agent
from shared.models import JSONValue, LLMMessage, ToolRequest, ToolResult
from tools.workspace_tools import WORKSPACE_ROOT


class DiffPreviewDialog(QDialog):
    def __init__(self, diff_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Предпросмотр патча")
        layout = QVBoxLayout()
        self.diff_view = QPlainTextEdit()
        self.diff_view.setReadOnly(True)
        mono = QFont("Courier New")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.diff_view.setFont(mono)
        self.diff_view.setPlainText(diff_text)
        layout.addWidget(self.diff_view)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)


class PatchDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Вставьте unified diff")
        layout = QVBoxLayout()
        self.editor = QPlainTextEdit()
        mono = QFont("Courier New")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.editor.setFont(mono)
        self.editor.setPlaceholderText("@@ -1,1 +1,1 @@\n-old\n+new\n")
        layout.addWidget(self.editor)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)

    def get_patch(self) -> str:
        return self.editor.toPlainText()


class LineNumberArea(QPlainTextEdit):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        mono = QFont("Courier New")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.setFont(mono)
        self.setReadOnly(True)
        self.setMaximumWidth(50)
        self.setVerticalScrollBarPolicy(self.verticalScrollBarPolicy())
        self.setHorizontalScrollBarPolicy(self.horizontalScrollBarPolicy())


class WorkspacePanel(QWidget):
    """Простой рабочий стол: дерево файлов + редактор + действия."""

    def __init__(
        self,
        agent: Agent,
        on_ask_ai: Callable[[str, str], None] | None = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.on_ask_ai = on_ask_ai
        self.current_path: Path | None = None
        self.original_text: str = ""

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderHidden(True)
        self.file_tree.itemClicked.connect(self._handle_item_clicked)

        self.line_numbers = LineNumberArea()
        self.editor = QPlainTextEdit()
        mono = QFont("Courier New")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        self.editor.setFont(mono)
        self.editor.textChanged.connect(self._update_line_numbers)
        self.editor.cursorPositionChanged.connect(self._update_workspace_context)
        self.editor.verticalScrollBar().valueChanged.connect(
            self.line_numbers.verticalScrollBar().setValue
        )

        self.status = QLabel()
        self.path_label = QLabel("Файл не выбран")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Поиск по дереву...")
        self.search_input.textChanged.connect(self._filter_tree)

        self.save_btn = QPushButton("💾 Сохранить")
        self.revert_btn = QPushButton("↩ Вернуть")
        self.patch_btn = QPushButton("🩹 Apply patch")
        self.ask_btn = QPushButton("❓ Спросить AI")
        self.run_btn = QPushButton("▶ Запустить")

        self.save_btn.clicked.connect(self._save_file)
        self.revert_btn.clicked.connect(self._revert_file)
        self.patch_btn.clicked.connect(self._apply_patch)
        self.ask_btn.clicked.connect(self._ask_ai)
        self.run_btn.clicked.connect(self._run_code)

        editor_layout = QHBoxLayout()
        editor_layout.addWidget(self.line_numbers)
        editor_layout.addWidget(self.editor)

        actions = QHBoxLayout()
        for btn in (self.save_btn, self.revert_btn, self.patch_btn, self.ask_btn, self.run_btn):
            actions.addWidget(btn)

        right = QVBoxLayout()
        right.addWidget(self.path_label)
        right.addLayout(editor_layout)
        right.addLayout(actions)
        right.addWidget(self.status)

        splitter = QSplitter()
        tree_container = QWidget()
        tree_layout = QVBoxLayout()
        tree_layout.addWidget(QLabel(f"Workspace: {WORKSPACE_ROOT}"))
        tree_layout.addWidget(self.search_input)
        tree_layout.addWidget(self.file_tree)
        tree_container.setLayout(tree_layout)
        splitter.addWidget(tree_container)

        editor_container = QWidget()
        editor_container.setLayout(right)
        splitter.addWidget(editor_container)

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

        self.refresh_tree()

    def refresh_tree(self) -> None:
        result = self._call_tool("workspace_list", {})
        if not result or not result.ok:
            self._set_status(result.error if result else "Ошибка загрузки дерева.")
            return
        tree_raw = result.data.get("tree")
        if not isinstance(tree_raw, list):
            self._set_status("Некорректный формат дерева файлов.")
            return
        self.file_tree.clear()
        for node in tree_raw:
            if not isinstance(node, dict):
                continue
            item = self._build_item(node)
            if item:
                self.file_tree.addTopLevelItem(item)
        self.file_tree.expandAll()
        self._set_status("Дерево файлов обновлено.")

    def _build_item(self, node: Mapping[str, JSONValue]) -> QTreeWidgetItem | None:
        node_type = node.get("type")
        name = str(node.get("name") or "")
        if not name:
            return None
        item = QTreeWidgetItem([name])
        if node_type == "dir":
            item.setData(0, Qt.ItemDataRole.UserRole, None)
            children = node.get("children") or []
            if isinstance(children, list):
                for child in children:
                    if not isinstance(child, dict):
                        continue
                    child_item = self._build_item(child)
                    if child_item:
                        item.addChild(child_item)
        elif node_type == "file":
            path = str(node.get("path") or "")
            item.setData(0, Qt.ItemDataRole.UserRole, path)
        return item

    def _filter_tree(self, text: str) -> None:
        text_lower = text.lower().strip()
        root_count = self.file_tree.topLevelItemCount()
        for i in range(root_count):
            item = self.file_tree.topLevelItem(i)
            if item:
                self._apply_filter(item, text_lower)

    def _apply_filter(self, item: QTreeWidgetItem, text_lower: str) -> bool:
        visible = False
        if text_lower in item.text(0).lower():
            visible = True
        for i in range(item.childCount()):
            child = item.child(i)
            child_visible = self._apply_filter(child, text_lower)
            visible = visible or child_visible
        item.setHidden(not visible)
        return visible

    def _handle_item_clicked(self, item: QTreeWidgetItem) -> None:
        path_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not path_data:
            return
        self._load_file(Path(str(path_data)))

    def _load_file(self, relative_path: Path) -> None:
        result = self._call_tool("workspace_read", {"path": str(relative_path)})
        if not result or not result.ok:
            self._set_status(result.error if result else "Ошибка чтения файла.")
            return
        content = str(result.data.get("output") or "")
        self.editor.setPlainText(content)
        self.current_path = relative_path
        self.original_text = content
        self.path_label.setText(f"Файл: {relative_path}")
        self._update_line_numbers()
        self._update_workspace_context()
        self._set_status("Файл открыт.")

    def _save_file(self) -> None:
        if not self.current_path:
            self._set_status("Нет выбранного файла.")
            return
        content = self.editor.toPlainText()
        result = self._call_tool(
            "workspace_write", {"path": str(self.current_path), "content": content}
        )
        if result and result.ok:
            self.original_text = content
            self._update_workspace_context()
            self._set_status("Файл сохранён.")
        else:
            self._set_status(result.error if result else "Ошибка сохранения.")

    def _revert_file(self) -> None:
        if not self.current_path:
            self._set_status("Нет выбранного файла.")
            return
        self._load_file(self.current_path)

    def _apply_patch(self) -> None:
        if not self.current_path:
            self._set_status("Нет выбранного файла.")
            return
        dialog = PatchDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        patch_text = dialog.get_patch()
        if not patch_text.strip():
            self._set_status("Пустой патч.")
            return

        dry_result = self._call_tool(
            "workspace_patch",
            {"path": str(self.current_path), "patch": patch_text, "dry_run": True},
        )
        if not dry_result or not dry_result.ok:
            self._set_status(dry_result.error if dry_result else "Ошибка dry-run.")
            return
        patched_content = str(dry_result.data.get("content") or "")
        diff = self._make_diff(self.editor.toPlainText(), patched_content)
        preview = DiffPreviewDialog(diff, self)
        if preview.exec() != QDialog.DialogCode.Accepted:
            self._set_status("Патч отклонён.")
            return

        apply_result = self._call_tool(
            "workspace_patch",
            {"path": str(self.current_path), "patch": patch_text, "dry_run": False},
        )
        if apply_result and apply_result.ok:
            self.editor.setPlainText(patched_content)
            self.original_text = patched_content
            self._update_workspace_context()
            self._set_status("Патч применён.")
        else:
            self._set_status(apply_result.error if apply_result else "Ошибка применения патча.")

    def _ask_ai(self) -> None:
        if not self.current_path:
            self._set_status("Нет выбранного файла.")
            return
        question, ok = QInputDialog.getText(self, "Вопрос к агенту", "Введите вопрос:", text="")
        if not ok or not question.strip():
            return
        selection = self._get_selection_text()
        self.agent.set_workspace_context(
            str(self.current_path), self.editor.toPlainText(), selection
        )
        reply = self.agent.respond([LLMMessage(role="user", content=question.strip())])
        self._set_status("Ответ получен, смотрите окно.")
        QMessageBox.information(self, "Ответ агента", reply)
        if self.on_ask_ai:
            self.on_ask_ai(question.strip(), reply)

    def _run_code(self) -> None:
        if not self.current_path:
            self._set_status("Нет выбранного файла.")
            return
        result = self._call_tool("workspace_run", {"path": str(self.current_path)})
        if not result:
            self._set_status("Ошибка запуска.")
            return
        if not result.ok:
            QMessageBox.warning(self, "Ошибка запуска", result.error or "Неизвестная ошибка")
            self._set_status(result.error or "Ошибка запуска.")
            return
        stdout = str(result.data.get("output") or "")
        stderr = str(result.data.get("stderr") or "")
        exit_code = result.data.get("exit_code")
        msg_lines = [f"exit_code: {exit_code}", f"stdout:\n{stdout}"]
        if stderr:
            msg_lines.append(f"stderr:\n{stderr}")
        QMessageBox.information(self, "Результат запуска", "\n\n".join(msg_lines))
        self._set_status("Скрипт выполнен.")

    def _update_line_numbers(self) -> None:
        lines = self.editor.blockCount()
        numbers = "\n".join(str(i) for i in range(1, lines + 1))
        self.line_numbers.setPlainText(numbers)
        self.line_numbers.verticalScrollBar().setValue(self.editor.verticalScrollBar().value())

    def _get_selection_text(self) -> str | None:
        cursor = self.editor.textCursor()
        if cursor.hasSelection():
            return cursor.selectedText().replace("\u2029", "\n")
        return None

    def _update_workspace_context(self) -> None:
        if not self.current_path:
            return
        selection = self._get_selection_text()
        self.agent.set_workspace_context(
            str(self.current_path), self.editor.toPlainText(), selection
        )

    def _call_tool(self, name: str, args: dict[str, JSONValue]) -> ToolResult | None:
        try:
            tool_args: dict[str, JSONValue] = {k: v for k, v in args.items() if v is not None}
            result = self.agent.tool_registry.call(ToolRequest(name=name, args=tool_args))
            return result
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Ошибка инструмента: {exc}")
            return None

    def _set_status(self, text: str | None) -> None:
        self.status.setText(text or "")

    def _make_diff(self, original: str, patched: str) -> str:
        diff_lines = difflib.unified_diff(
            original.splitlines(),
            patched.splitlines(),
            fromfile="original",
            tofile="patched",
            lineterm="",
        )
        return "\n".join(diff_lines)
