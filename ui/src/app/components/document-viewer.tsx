import React, { useState, useRef, useEffect } from "react";
import {
  X,
  Copy,
  FileText,
  Check,
  ArrowLeft,
  Download,
  ChevronDown,
} from "lucide-react";

export interface DocumentViewerProps {
  title?: string;
  type?: string;
  content?: string;
  onBack?: () => void;
  onClose?: () => void;
  className?: string;
}

// Formats available for document types vs code types
const documentFormats = ["MD", "TXT", "PDF", "HTML"];
const codeExtensions = ["PY", "JS", "TS", "JSX", "TSX", "CSS", "JSON", "HTML", "SQL", "SH", "YAML", "YML", "TOML"];

function isCodeFile(type: string): boolean {
  return codeExtensions.includes(type.toUpperCase());
}

function getMimeType(format: string): string {
  switch (format.toUpperCase()) {
    case "PDF": return "application/pdf";
    case "HTML": return "text/html";
    case "JSON": return "application/json";
    case "MD": return "text/markdown";
    default: return "text/plain";
  }
}

function normalizeLanguage(language: string, fileType: string): string {
  const raw = language.trim().toLowerCase();
  if (raw) {
    if (raw === "javascript") return "js";
    if (raw === "typescript") return "ts";
    if (raw === "python") return "py";
    if (raw === "shell" || raw === "bash" || raw === "zsh") return "sh";
    if (raw === "yml") return "yaml";
    return raw;
  }
  const fallback = fileType.trim().toLowerCase();
  if (fallback === "javascript") return "js";
  if (fallback === "typescript") return "ts";
  if (fallback === "python") return "py";
  return fallback;
}

function keywordSet(language: string): Set<string> {
  if (language === "py") {
    return new Set([
      "class", "def", "import", "from", "return", "if", "else", "elif", "for", "while", "in",
      "not", "and", "or", "True", "False", "None", "try", "except", "finally", "raise", "with",
      "as", "lambda", "pass", "break", "continue", "yield", "global", "nonlocal", "assert",
    ]);
  }
  if (language === "json" || language === "yaml" || language === "toml") {
    return new Set(["true", "false", "null"]);
  }
  if (language === "sh") {
    return new Set([
      "if", "then", "else", "fi", "for", "in", "do", "done", "case", "esac", "while", "function",
      "export", "local", "return", "echo", "cd", "pwd", "test",
    ]);
  }
  return new Set([
    "const", "let", "var", "function", "return", "if", "else", "for", "while", "import", "from",
    "export", "default", "class", "new", "true", "false", "null", "undefined", "try", "catch",
    "finally", "throw", "await", "async", "switch", "case", "break", "continue", "typeof", "in",
    "instanceof", "extends", "implements", "interface", "type",
  ]);
}

function highlightToken(
  token: string,
  keywords: Set<string>,
  tokenIndex: number,
  keyPrefix: string,
): React.ReactNode {
  if (!token) {
    return null;
  }
  if (keywords.has(token)) {
    return <span key={`${keyPrefix}-kw-${tokenIndex}`} className="text-[#c792ea]">{token}</span>;
  }
  if (/^["'`].*["'`]$/.test(token) || token.startsWith("'") || token.startsWith('"')) {
    return <span key={`${keyPrefix}-str-${tokenIndex}`} className="text-[#c3e88d]">{token}</span>;
  }
  if (/^\d+(\.\d+)?$/.test(token)) {
    return <span key={`${keyPrefix}-num-${tokenIndex}`} className="text-[#f78c6c]">{token}</span>;
  }
  if (/^[{}()[\].,:;=+\-*/<>!&|]+$/.test(token)) {
    return <span key={`${keyPrefix}-sym-${tokenIndex}`} className="text-[#89ddff]">{token}</span>;
  }
  return <span key={`${keyPrefix}-txt-${tokenIndex}`} className="text-[#cfd8dc]">{token}</span>;
}

function renderHighlightedCodeBlock(
  code: string,
  language: string,
  keyPrefix: string,
): React.ReactNode {
  const lang = normalizeLanguage(language, language);
  const keywords = keywordSet(lang);
  const lines = code.replace(/\r\n/g, "\n").split("\n");
  return (
    <div key={`${keyPrefix}-code`} className="my-3 overflow-hidden rounded-lg border border-[#1f1f24] bg-[#0d0d12]">
      <div className="border-b border-[#1f1f24] bg-[#111118] px-3 py-1.5 text-[11px] uppercase tracking-wide text-[#7f8a9a]">
        {lang || "text"}
      </div>
      <pre className="overflow-x-auto px-3 py-3 text-[13px] leading-relaxed font-mono">
        {lines.map((line, lineIndex) => {
          const commentMarker = line.indexOf("//");
          const hashCommentMarker = line.indexOf("#");
          let cut = -1;
          if (commentMarker >= 0) cut = commentMarker;
          if (hashCommentMarker >= 0) cut = cut >= 0 ? Math.min(cut, hashCommentMarker) : hashCommentMarker;
          const codePart = cut >= 0 ? line.slice(0, cut) : line;
          const commentPart = cut >= 0 ? line.slice(cut) : "";
          const tokens = codePart.split(/(\s+|[{}()[\].,:;=+\-*/<>!&|]+)/).filter((token) => token.length > 0);
          return (
            <div key={`${keyPrefix}-line-${lineIndex}`} className="flex">
              <span className="mr-4 inline-block w-7 select-none text-right text-[#5a6578]">
                {lineIndex + 1}
              </span>
              <span>
                {tokens.map((token, tokenIndex) => highlightToken(token, keywords, tokenIndex, `${keyPrefix}-${lineIndex}`))}
                {commentPart ? <span className="text-[#6a9955]">{commentPart}</span> : null}
              </span>
            </div>
          );
        })}
      </pre>
    </div>
  );
}

export function DocumentViewer({
  title = "Artifact",
  type = "MD",
  content = "",
  onBack,
  onClose,
  className = "",
}: DocumentViewerProps) {
  const [copied, setCopied] = useState(false);
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);
  const downloadRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (downloadRef.current && !downloadRef.current.contains(e.target as Node)) {
        setShowDownloadMenu(false);
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = (format: string) => {
    const blob = new Blob([content], { type: getMimeType(format) });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${title}.${format.toLowerCase()}`;
    a.click();
    URL.revokeObjectURL(url);
    setShowDownloadMenu(false);
  };

  const downloadFormats = isCodeFile(type) ? [type.toUpperCase()] : documentFormats;

  // Simple markdown renderer
  const renderMarkdown = (md: string) => {
    const lines = md.split("\n");
    const elements: React.ReactNode[] = [];
    let inTable = false;
    let tableHeaders: string[] = [];
    let tableRows: string[][] = [];
    let inList = false;
    let listItems: React.ReactNode[] = [];
    let inCodeFence = false;
    let codeFenceLanguage = "";
    let codeFenceLines: string[] = [];

    const flushTable = () => {
      if (tableHeaders.length > 0) {
        elements.push(
          <div key={`table-${elements.length}`} className="my-3 overflow-x-auto">
            <table className="w-full text-[13px] border-collapse">
              <thead>
                <tr>
                  {tableHeaders.map((h, i) => (
                    <th
                      key={i}
                      className="text-left px-3 py-2 bg-[#141418] text-[#aaa] border border-[#1f1f24]"
                    >
                      {h.trim()}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableRows.map((row, ri) => (
                  <tr key={ri}>
                    {row.map((cell, ci) => (
                      <td
                        key={ci}
                        className="px-3 py-2 text-[#c0c0c0] border border-[#1f1f24]"
                      >
                        {cell.trim()}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      }
      tableHeaders = [];
      tableRows = [];
      inTable = false;
    };

    const flushList = () => {
      if (listItems.length > 0) {
        elements.push(
          <ul
            key={`list-${elements.length}`}
            className="space-y-1 my-2 ml-1"
          >
            {listItems}
          </ul>
        );
      }
      listItems = [];
      inList = false;
    };

    const flushCodeFence = () => {
      if (!inCodeFence) {
        return;
      }
      elements.push(
        renderHighlightedCodeBlock(
          codeFenceLines.join("\n"),
          codeFenceLanguage,
          `fence-${elements.length}`,
        ),
      );
      inCodeFence = false;
      codeFenceLanguage = "";
      codeFenceLines = [];
    };

    const renderInline = (text: string) => {
      // Code backticks
      const parts = text.split(/(`[^`]+`)/);
      return parts.map((part, i) => {
        if (part.startsWith("`") && part.endsWith("`")) {
          return (
            <code
              key={i}
              className="px-1.5 py-0.5 rounded bg-[#2a2a2e] text-[#e8a0bf] text-[12px] font-mono"
            >
              {part.slice(1, -1)}
            </code>
          );
        }
        // Bold
        const boldParts = part.split(/(\*\*[^*]+\*\*)/);
        return boldParts.map((bp, j) => {
          if (bp.startsWith("**") && bp.endsWith("**")) {
            return (
              <span key={`${i}-${j}`} className="text-[#e0e0e0]">
                {bp.slice(2, -2)}
              </span>
            );
          }
          return <span key={`${i}-${j}`}>{bp}</span>;
        });
      });
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      if (trimmed.startsWith("```")) {
        if (inCodeFence) {
          flushCodeFence();
        } else {
          flushTable();
          flushList();
          inCodeFence = true;
          codeFenceLanguage = trimmed.slice(3).trim();
          codeFenceLines = [];
        }
        continue;
      }

      if (inCodeFence) {
        codeFenceLines.push(line);
        continue;
      }

      // Table detection
      if (line.includes("|") && trimmed.startsWith("|")) {
        const cells = line
          .split("|")
          .filter((c) => c.trim() !== "")
          .map((c) => c.trim());

        if (cells.every((c) => /^[-:]+$/.test(c))) {
          // Separator line, skip
          continue;
        }

        if (!inTable) {
          flushList();
          inTable = true;
          tableHeaders = cells;
        } else {
          tableRows.push(cells);
        }
        continue;
      } else if (inTable) {
        flushTable();
      }

      // List items
      if (trimmed.startsWith("- ")) {
        if (!inList) inList = true;
        listItems.push(
          <li
            key={`li-${i}`}
            className="flex items-start gap-2 text-[13px] text-[#b0b0b0]"
          >
            <span className="text-[#6366f1] mt-1">-</span>
            <span>{renderInline(trimmed.slice(2))}</span>
          </li>
        );
        continue;
      } else if (inList) {
        flushList();
      }

      // Horizontal rule
      if (trimmed === "---") {
        elements.push(
            <hr
              key={`hr-${i}`}
              className="border-[#1f1f24] my-4"
            />
        );
        continue;
      }

      // Headings
      if (line.startsWith("## ")) {
        elements.push(
          <h2
            key={`h2-${i}`}
            className="text-[16px] text-[#e0e0e0] mt-5 mb-2"
          >
            {line.slice(3)}
          </h2>
        );
        continue;
      }

      if (line.startsWith("### ")) {
        elements.push(
          <h3
            key={`h3-${i}`}
            className="text-[14px] text-[#ccc] mt-4 mb-1.5"
          >
            {line.slice(4)}
          </h3>
        );
        continue;
      }

      // Empty line
      if (trimmed === "") {
        elements.push(<div key={`sp-${i}`} className="h-2" />);
        continue;
      }

      // Regular text
      elements.push(
        <p key={`p-${i}`} className="text-[13px] text-[#b0b0b0] leading-relaxed">
          {renderInline(line)}
        </p>
      );
    }

    // Flush remaining
    if (inTable) flushTable();
    if (inList) flushList();
    if (inCodeFence) flushCodeFence();

    return elements;
  };

  return (
    <div
      className={`flex flex-col h-full bg-[#0b0b0d] ${className}`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-[#111115]">
        <div className="flex items-center gap-3">
          {onBack && (
            <button
              onClick={onBack}
              className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[13px] text-[#888] hover:text-[#ddd] hover:bg-[#1b1b20] transition-all cursor-pointer"
            >
              <ArrowLeft className="w-4 h-4" />
              Back
            </button>
          )}
          <div className="flex items-center gap-2">
            <FileText className="w-4 h-4 text-[#888]" />
            <span className="text-[13px] text-[#ccc]">{title}</span>
            <span className="text-[11px] text-[#666] bg-[#1b1b20] px-1.5 py-0.5 rounded">
              {type}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[12px] text-[#888] hover:text-[#ddd] hover:bg-[#1b1b20] transition-all cursor-pointer"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-green-400" />
                <span className="text-green-400">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                Copy
              </>
            )}
          </button>
          <div
            ref={downloadRef}
            className="relative"
          >
            <button
              onClick={() => setShowDownloadMenu(!showDownloadMenu)}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[12px] text-[#888] hover:text-[#ddd] hover:bg-[#1b1b20] transition-all cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              Download
              <ChevronDown className="w-3.5 h-3.5" />
            </button>
            {showDownloadMenu && (
              <div
                className="absolute right-0 top-full mt-1 rounded-lg bg-[#141418] border border-[#1f1f24] shadow-xl shadow-black/40 py-1 min-w-[120px] z-10"
              >
                {downloadFormats.map((format) => (
                  <button
                    key={format}
                    onClick={() => handleDownload(format)}
                    className="flex items-center gap-2 w-full px-3 py-2 text-[12px] text-[#ccc] hover:bg-[#1b1b20] transition-colors cursor-pointer"
                  >
                    <Download className="w-3 h-3 text-[#666]" />
                    .{format.toLowerCase()}
                  </button>
                ))}
              </div>
            )}
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-[#555] hover:text-[#ccc] transition-colors p-1 cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto" data-scrollbar="auto">
        <div className="px-6 py-5">
          {content ? (
            isCodeFile(type) && !content.includes("```")
              ? renderHighlightedCodeBlock(content, type, "raw-code")
              : renderMarkdown(content)
          ) : (
            <p className="text-[13px] text-[#555]">Нет содержимого.</p>
          )}
        </div>
      </div>
    </div>
  );
}
