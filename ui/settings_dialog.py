from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from config.shell_config import ShellConfig, load_shell_config, save_shell_config
from core.agent import Agent
from llm.types import ModelConfig


class SettingsDialog(QDialog):
    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.agent = agent
        self.setWindowTitle("Настройки SlavikAI")

        self.main_provider = QComboBox()
        self.main_provider.addItems(["openrouter", "local"])
        self.main_model = QLineEdit()
        self.main_api_key = QLineEdit()
        self.main_base_url = QLineEdit()

        self.critic_enabled = QCheckBox("Включить критика (Dual)")
        self.critic_provider = QComboBox()
        self.critic_provider.addItems(["openrouter", "local"])
        self.critic_model = QLineEdit()
        self.critic_api_key = QLineEdit()
        self.critic_base_url = QLineEdit()

        self.shell_config_path = QLineEdit()
        self.shell_allowed = QPlainTextEdit()
        self.shell_timeout = QSpinBox()
        self.shell_max_output = QSpinBox()
        self.shell_sandbox = QLineEdit()

        self.mode_select = QComboBox()
        self.mode_select.addItems(["single", "dual", "critic-only"])

        self._init_state()
        self._build_layout()

    def _init_state(self) -> None:
        main_cfg: ModelConfig | None = getattr(self.agent, "main_config", None)
        critic_cfg: ModelConfig | None = getattr(self.agent, "critic_config", None)
        if main_cfg:
            self.main_provider.setCurrentText(main_cfg.provider)
            self.main_model.setText(main_cfg.model)
            self.main_base_url.setText(main_cfg.base_url or "")
        else:
            self.main_provider.setCurrentIndex(0)
            self.main_model.setText("")
            self.main_base_url.setText("")
        self.main_api_key.setText(getattr(self.agent, "main_api_key", "") or "")

        if critic_cfg:
            self.critic_enabled.setChecked(True)
            self.critic_provider.setCurrentText(critic_cfg.provider)
            self.critic_model.setText(critic_cfg.model)
            self.critic_base_url.setText(critic_cfg.base_url or "")
            self.critic_api_key.setText(getattr(self.agent, "critic_api_key", "") or "")
            self.mode_select.setCurrentText("dual")
        else:
            self.critic_enabled.setChecked(False)
            self.critic_model.setText("")
            self.critic_provider.setCurrentIndex(0)
            self.mode_select.setCurrentText("single")

        cfg_path = getattr(self.agent, "shell_config_path", "config/shell_config.json")
        self.shell_config_path.setText(cfg_path)
        self._init_shell_config(Path(cfg_path))

    def _init_shell_config(self, path: Path) -> None:
        try:
            cfg = load_shell_config(path)
        except Exception:
            cfg = ShellConfig()
        self.shell_allowed.setPlainText("\n".join(cfg.allowed_commands))
        self.shell_timeout.setRange(1, 300)
        self.shell_timeout.setValue(cfg.timeout_seconds)
        self.shell_max_output.setRange(100, 100_000)
        self.shell_max_output.setValue(cfg.max_output_chars)
        self.shell_sandbox.setText(cfg.sandbox_root)

    def _build_layout(self) -> None:
        layout = QVBoxLayout()

        # Основная модель
        layout.addWidget(QLabel("🧠 Основная модель"))
        main_form = QFormLayout()
        main_form.addRow("Провайдер", self.main_provider)
        main_form.addRow("Model ID", self.main_model)
        main_form.addRow("API Key", self.main_api_key)
        main_form.addRow("Base URL (для local)", self.main_base_url)
        layout.addLayout(main_form)

        # Критик
        layout.addWidget(self.critic_enabled)
        critic_form = QFormLayout()
        critic_form.addRow("Провайдер", self.critic_provider)
        critic_form.addRow("Model ID", self.critic_model)
        critic_form.addRow("API Key", self.critic_api_key)
        critic_form.addRow("Base URL (для local)", self.critic_base_url)
        layout.addLayout(critic_form)

        # Режим DualBrain
        layout.addWidget(QLabel("Режим работы моделей"))
        layout.addWidget(self.mode_select)

        # Shell config
        layout.addWidget(QLabel("Shell tool"))
        shell_form = QFormLayout()
        shell_form.addRow("Config path", self.shell_config_path)
        shell_form.addRow("Allowed commands (по строке)", self.shell_allowed)
        shell_form.addRow("Timeout (сек)", self.shell_timeout)
        shell_form.addRow("Max output chars", self.shell_max_output)
        shell_form.addRow("Sandbox root", self.shell_sandbox)
        layout.addLayout(shell_form)

        # Кнопки
        btns = QHBoxLayout()
        self.save_btn = QPushButton("✅ Применить")
        self.save_btn.clicked.connect(self.apply_settings)
        cancel_btn = QPushButton("Отмена")
        cancel_btn.clicked.connect(self.close)
        btns.addWidget(self.save_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        self.setLayout(layout)

    def apply_settings(self) -> None:
        model_text = self.main_model.text().strip()
        if not model_text:
            QMessageBox.warning(self, "Модель не выбрана", "Укажите model id основной модели.")
            return

        main_cfg = self._build_config(
            base=self.agent.main_config if hasattr(self.agent, "main_config") else None,
            provider=self.main_provider.currentText(),
            model=model_text,
            base_url=self.main_base_url.text().strip(),
        )
        critic_cfg = None
        critic_key = None
        if self.critic_enabled.isChecked():
            critic_model = self.critic_model.text().strip()
            if not critic_model:
                QMessageBox.warning(
                    self,
                    "Критик не выбран",
                    "Укажите model id критика или выключите его.",
                )
                return
            critic_cfg = self._build_config(
                base=self.agent.critic_config if hasattr(self.agent, "critic_config") else None,
                provider=self.critic_provider.currentText(),
                model=critic_model,
                base_url=self.critic_base_url.text().strip(),
            )
            critic_key = self.critic_api_key.text().strip() or None

        main_key = self.main_api_key.text().strip() or None

        self.agent.reconfigure_models(
            main_config=main_cfg,
            critic_config=critic_cfg,
            main_api_key=main_key,
            critic_api_key=critic_key,
        )

        mode = self.mode_select.currentText()
        if critic_cfg is None and mode != "single":
            mode = "single"
        self.agent.set_mode(mode)

        # Shell config save
        try:
            shell_cfg = self._build_shell_config()
            cfg_path = Path(self.shell_config_path.text().strip() or "config/shell_config.json")
            save_shell_config(shell_cfg, cfg_path)
            if hasattr(self.agent, "shell_config_path"):
                self.agent.shell_config_path = str(cfg_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Ошибка shell config",
                f"Не удалось сохранить shell config: {exc}",
            )
            return
        self.close()

    def _build_config(
        self, base: ModelConfig | None, provider: str, model: str, base_url: str | None
    ) -> ModelConfig:
        return ModelConfig(
            provider=provider,  # type: ignore[arg-type]
            model=model,
            temperature=base.temperature if base else 0.7,
            top_p=base.top_p if base else None,
            max_tokens=base.max_tokens if base else None,
            base_url=base_url or None,
            api_key=base.api_key if base else None,
            extra_headers=base.extra_headers if base else {},
            system_prompt=base.system_prompt if base else None,
            mode=base.mode if base else "default",
        )

    def _build_shell_config(self) -> ShellConfig:
        raw_allowed = self.shell_allowed.toPlainText().splitlines()
        allowed = [cmd.strip() for line in raw_allowed for cmd in line.split(",") if cmd.strip()]
        return ShellConfig(
            allowed_commands=allowed or ShellConfig().allowed_commands,
            timeout_seconds=self.shell_timeout.value(),
            max_output_chars=self.shell_max_output.value(),
            sandbox_root=self.shell_sandbox.text().strip() or ShellConfig().sandbox_root,
        )
