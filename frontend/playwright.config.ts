import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL(".", import.meta.url)), "..");
const port = 8765;

export default defineConfig({
  testDir: "./tests",
  timeout: 45_000,
  retries: 1,
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    browserName: "chromium",
    permissions: ["microphone"],
    launchOptions: {
      args: [
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        `--use-file-for-fake-audio-capture=${resolve(root, "samples/question.wav")}`,
      ],
    },
  },
  webServer: {
    command: `${process.env.GATEWAY_PYTHON ?? "python3"} -m gateway.app`,
    cwd: root,
    url: `http://127.0.0.1:${port}/api/health`,
    reuseExistingServer: false,
    timeout: 30_000,
    env: {
      ADAPTER_MODE: "mock",
      PYTHONPATH: resolve(root, "gateway"),
      BIND_HOST: "127.0.0.1",
      BIND_PORT: String(port),
      STUN_URLS: "",
      AUTH_TOKEN: "",
    },
  },
});
