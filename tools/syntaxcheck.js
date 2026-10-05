// Structural sanity check on the Python sources: strip strings and comments,
// then verify brackets balance. Not a substitute for a real parse, but it
// catches the truncation and stray-comma mistakes a transliteration hides.
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const FILES = [
  "eq_layer/policies.py",
  "eq_layer/affect.py",
  "eq_layer/steer.py",
  "eq_layer/__init__.py",
  "eval/run.py",
];

let bad = 0;

for (const rel of FILES) {
  const src = fs.readFileSync(path.join(ROOT, rel), "utf8");
  const stripped = src
    .replace(/"""[\s\S]*?"""/g, 'T')
    .replace(/'''[\s\S]*?'''/g, "T")
    .replace(/(^|\s)("""|''')[\s\S]*?\2/g, "$1T")
    .replace(/(^|\s)#[^\n]*/g, "$1")
    .replace(/"(?:\\.|[^"\\\n])*"/g, 'S')
    .replace(/'(?:\\.|[^'\\\n])*'/g, "S");

  const pairs = { "(": ")", "[": "]", "{": "}" };
  const opens = Object.keys(pairs);
  const stack = [];
  let ok = true;

  for (const ch of stripped) {
    if (opens.includes(ch)) stack.push(ch);
    else if (Object.values(pairs).includes(ch)) {
      const top = stack.pop();
      if (top && pairs[top] !== ch) ok = false;
    }
  }
  if (stack.length) ok = false;

  // A trailing comma before any closing bracket is legal Python, so it is not
  // flagged. Only separators with nothing between them are a real typo.
  const emptySep = /,\s*,/.test(stripped) || /\(\s*,/.test(stripped) || /\[\s*,/.test(stripped);
  if (!ok || emptySep) bad++;
  console.log(
    `${rel.padEnd(24)} ${ok ? "balanced" : "UNBALANCED"}${emptySep ? "  EMPTY SEPARATOR" : ""}`
  );
}

console.log(bad ? `\n${bad} file(s) need a real parse` : "\nstructure ok (still needs python to compile)");
process.exit(bad ? 1 : 0);