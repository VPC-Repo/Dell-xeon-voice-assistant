// Copy index.html into dist with a content-hash query on the app.js
// reference, so browsers can never pair a fresh page with a stale cached
// bundle (the mismatch leaves the UI dead: the old script crashes on
// elements that no longer exist).
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";

const hash = createHash("sha256")
  .update(readFileSync("dist/app.js"))
  .digest("hex")
  .slice(0, 10);
const html = readFileSync("public/index.html", "utf8").replace(
  'src="app.js"',
  `src="app.js?v=${hash}"`,
);
if (!html.includes(`app.js?v=${hash}`)) {
  throw new Error('index.html has no src="app.js" script tag to stamp');
}
writeFileSync("dist/index.html", html);
console.log(`stamped dist/index.html -> app.js?v=${hash}`);
