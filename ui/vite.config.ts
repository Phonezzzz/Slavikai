import path from "path";

import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/ui/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  assetsInclude: ["**/*.svg", "**/*.csv"],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) {
            return undefined;
          }

          if (
            id.includes("node_modules/react/") ||
            id.includes("node_modules/react-dom/")
          ) {
            return "vendor-react";
          }

          if (
            id.includes("node_modules/react-markdown/") ||
            id.includes("node_modules/remark-gfm/")
          ) {
            return "vendor-markdown";
          }

          if (id.includes("node_modules/lucide-react/")) {
            return "vendor-icons";
          }

          if (id.includes("node_modules/motion/")) {
            return "vendor-motion";
          }

          if (id.includes("node_modules/jszip/")) {
            return "vendor-zip";
          }

          if (id.includes("node_modules/@radix-ui/")) {
            return "vendor-radix";
          }

          if (
            id.includes("node_modules/clsx/") ||
            id.includes("node_modules/tailwind-merge/")
          ) {
            return "vendor-styles";
          }

          return "vendor";
        },
      },
    },
  },
});
