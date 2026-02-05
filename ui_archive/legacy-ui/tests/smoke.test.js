import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");
const modelConfigPath = path.join(repoRoot, "config", "model_config.json");

const baseUrl = process.env.SMOKE_BASE_URL || "http://127.0.0.1:8000";
const timeoutMs = Number(process.env.SMOKE_TIMEOUT_MS || "120000");
const xaiApiKey = (process.env.XAI_API_KEY || "").trim();

if (!xaiApiKey) {
  console.error("SMOKE FAILED: XAI_API_KEY не задан.");
  process.exit(1);
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const waitForHttp = async (url, timeout) => {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // keep polling
    }
    await sleep(500);
  }
  throw new Error(`Таймаут ожидания HTTP ${url}`);
};

const fetchFirstXaiModel = async () => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch("https://api.x.ai/v1/models", {
      headers: {
        Authorization: `Bearer ${xaiApiKey}`,
      },
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`xAI models HTTP ${response.status}`);
    }
    const payload = await response.json();
    if (!payload || typeof payload !== "object" || !Array.isArray(payload.data)) {
      throw new Error("Некорректный ответ xAI /models.");
    }
    const found = payload.data.find(
      (item) => item && typeof item === "object" && typeof item.id === "string" && item.id.trim()
    );
    if (!found) {
      throw new Error("xAI /models вернул пустой список.");
    }
    return found.id;
  } finally {
    clearTimeout(timeout);
  }
};

const prepareModelConfig = (modelId) => {
  let backup = null;
  if (fs.existsSync(modelConfigPath)) {
    backup = fs.readFileSync(modelConfigPath, "utf-8");
  }

  const payload = {
    main: {
      provider: "xai",
      model: modelId,
      temperature: 0.7,
    },
  };
  fs.writeFileSync(modelConfigPath, `${JSON.stringify(payload, null, 2)}\n`, "utf-8");

  return () => {
    if (backup === null) {
      try {
        fs.unlinkSync(modelConfigPath);
      } catch {
        // ignore
      }
      return;
    }
    fs.writeFileSync(modelConfigPath, backup, "utf-8");
  };
};

const startServer = () => {
  const pythonPath = path.join(repoRoot, "venv", "bin", "python");
  const child = spawn(pythonPath, ["-m", "server"], {
    cwd: repoRoot,
    env: {
      ...process.env,
      XAI_API_KEY: xaiApiKey,
      SLAVIK_MODEL_WHITELIST: "*",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });

  child.stdout.on("data", (chunk) => process.stdout.write(`[server] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[server] ${chunk}`));
  return child;
};

const ensureModelSelected = async (page) => {
  const selects = page.locator("select");
  const providerSelect = selects.nth(0);
  const modelSelect = selects.nth(1);

  await providerSelect.selectOption("xai");
  await page.waitForFunction(() => {
    const model = document.querySelectorAll("select")[1];
    if (!(model instanceof HTMLSelectElement)) {
      return false;
    }
    return Array.from(model.options).some((item) => item.value.trim().length > 0);
  });

  const modelValue = await modelSelect.evaluate((node) => {
    if (!(node instanceof HTMLSelectElement)) {
      return "";
    }
    const first = Array.from(node.options).find((item) => item.value.trim().length > 0);
    return first ? first.value : "";
  });

  if (!modelValue) {
    throw new Error("xAI модели не загрузились в UI.");
  }

  await modelSelect.selectOption(modelValue);
  await page.getByRole("button", { name: "Set", exact: true }).click();
  await page.waitForTimeout(300);
};

const runSmoke = async () => {
  const initialModel = await fetchFirstXaiModel();
  const restoreModelConfig = prepareModelConfig(initialModel);

  const server = startServer();
  let browser;
  try {
    await waitForHttp(`${baseUrl}/ui/api/status`, 30000);

    browser = await chromium.launch();
    const page = await browser.newPage();
    page.setDefaultTimeout(timeoutMs);

    await page.goto(`${baseUrl}/ui/`, { waitUntil: "domcontentloaded" });
    await page.getByText("Conversation", { exact: false }).first().waitFor();

    await ensureModelSelected(page);

    const prompt = `smoke ping ${Date.now()}`;
    await page.getByPlaceholder("Send a message").fill(prompt);
    await page.getByRole("button", { name: "Send" }).click();

    await page.getByText(prompt, { exact: false }).first().waitFor();
    await page.getByText("Assistant", { exact: true }).first().waitFor();

    console.log("SMOKE PASSED: UI отправляет сообщение и получает ответ.");
  } finally {
    restoreModelConfig();
    if (browser) {
      await browser.close();
    }
    if (!server.killed) {
      server.kill("SIGTERM");
      await sleep(300);
      if (!server.killed) {
        server.kill("SIGKILL");
      }
    }
  }
};

runSmoke().catch((error) => {
  console.error("SMOKE FAILED:", error instanceof Error ? error.message : String(error));
  process.exit(1);
});
