import { expect, test } from "@playwright/test";

test("connects through WebRTC and plays a mock response", async ({ page }) => {
  await page.addInitScript(() => {
    const original = AudioContext.prototype.createMediaStreamSource;
    (window as any).monitoredStreams = 0;
    AudioContext.prototype.createMediaStreamSource = function (stream) {
      (window as any).monitoredStreams += 1;
      return original.call(this, stream);
    };
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Open voice settings" }).click();
  await page.getByRole("checkbox", { name: "Show live captions" }).check();
  await page.getByRole("button", { name: "Close settings" }).click();

  await page.getByRole("button", { name: "Click to connect" }).click();
  await expect(page.getByRole("button", { name: "End conversation" })).toBeEnabled();
  await expect(page.locator("#transcript .line.user").first()).toContainText(
    "Tell me what you can do.",
    { timeout: 30_000 },
  );
  await expect(page.locator("#transcript .line.assistant").first()).toContainText(
    "mock assistant",
    { timeout: 15_000 },
  );
  await expect(page.locator("#player")).toHaveJSProperty("paused", false);
  // Both microphone and incoming speech must be monitored on the first call.
  await expect.poll(() => page.evaluate(() => (window as any).monitoredStreams)).toBe(2);
  await page.getByRole("button", { name: "Mute microphone", exact: true }).click();
  await page.getByRole("button", { name: "End conversation" }).click();
  await expect(page.locator("#player")).toHaveJSProperty("srcObject", null);
  await expect(page.getByRole("button", { name: "Mute microphone", exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Click to connect" })).toBeVisible();
});
