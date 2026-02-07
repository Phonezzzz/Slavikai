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

      // Table detection
      if (line.includes("|") && line.trim().startsWith("|")) {
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
      if (line.trim().startsWith("- ")) {
        if (!inList) inList = true;
        listItems.push(
          <li
            key={`li-${i}`}
            className="flex items-start gap-2 text-[13px] text-[#b0b0b0]"
          >
            <span className="text-[#6366f1] mt-1">-</span>
            <span>{renderInline(line.trim().slice(2))}</span>
          </li>
        );
        continue;
      } else if (inList) {
        flushList();
      }

      // Horizontal rule
      if (line.trim() === "---") {
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
      if (line.trim() === "") {
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
            renderMarkdown(content)
          ) : (
            <p className="text-[13px] text-[#555]">Нет содержимого.</p>
          )}
        </div>
      </div>
    </div>
  );
}
