// Translate eq_layer to JS and diff the selection table against eval/cases.jsonl.
//
// Necessary because this machine has no working Python interpreter, so
// eval/run.py cannot be executed here. It reads the policy table and keyword
// sets straight out of the Python source, so the two cannot drift on the
// values that matter — though it does not check the Python syntax itself.
//
// Usage: node tools/crosscheck.js

const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const py = (rel) => fs.readFileSync(path.join(ROOT, rel), "utf8");

// --- read the policy table straight out of the Python source -------------

const policiesSrc = py("eq_layer/policies.py");
const affectSrc = py("eq_layer/affect.py");

function extractTuple(src, marker, names) {
  const start = src.indexOf(marker);
  if (start < 0) throw new Error(`marker not found: ${marker}`);
  let depth = 0, i = src.indexOf("(", start);
  const from = i;
  for (; i < src.length; i++) {
    if (src[i] === "(") depth++;
    else if (src[i] === ")") { depth--; if (depth === 0) break; }
  }
  const body = src.slice(from, i + 1);
  const out = {};
  for (const name of names) {
    const re = new RegExp(`name="${name}",[\\s\\S]*?preconditions=\\(([^)]*)\\)`);
    const m = body.match(re);
    if (!m) throw new Error(`policy not found or has no literal preconditions: ${name}`);
    out[name] = m[1].match(/"([^"]+)"/g).map((s) => s.slice(1, -1));
  }
  return out;
}

const POLICY_NAMES = [
  "mirror_specific",
  "validate_then_redirect",
  "deflate_tension",
  "direct_no_padding",
  "ask_one_question",
  "repair_hold_position",
  "repair_interrogation",
  "hold",
  "boundary",
  "hold_sustained_escalation",
];

const policies = extractTuple(policiesSrc, "POLICIES: tuple[Policy, ...] = (", POLICY_NAMES);

function extractStringTuple(src, marker) {
  const start = src.indexOf(marker);
  if (start < 0) throw new Error(`marker not found: ${marker}`);
  const rest = src.slice(start);
  const body = rest.slice(rest.indexOf("("));
  let depth = 0, end = 0;
  for (let i = 0; i < body.length; i++) {
    if (body[i] === "(") depth++;
    else if (body[i] === ")") { depth--; if (depth === 0) { end = i; break; } }
  }
  return body.slice(0, end + 1).match(/"([^"]+)"/g).map((s) => s.slice(1, -1));
}

const markers = {
  ESCALATION: extractStringTuple(affectSrc, "ESCALATION_MARKERS = ("),
  SOFTENERS: extractStringTuple(affectSrc, "SOFTENERS = ("),
  CORRECTION: extractStringTuple(affectSrc, "CORRECTION_MARKERS = ("),
  DISCLOSURE: extractStringTuple(affectSrc, "DISCLOSURE_MARKERS = ("),
  QUESTION_OPENERS: extractStringTuple(affectSrc, "QUESTION_OPENERS = ("),
};

const SUBTEXTS = extractStringTuple(policiesSrc, "SUBTEXTS = (");
const PUNCT = "?!.,;:";

// --- transliteration -----------------------------------------------------

const r3 = (n) => Math.round(n * 1000) / 1000;

function arousal(text) {
  if (!text) return 0;
  const low = text.toLowerCase().trim();
  let hits = 0, soft = 0;
  for (const m of markers.ESCALATION) if (low.includes(m)) hits++;
  for (const s of markers.SOFTENERS) if (low.includes(s)) soft++;
  let score = Math.min(1, 0.25 + hits * 0.22 - soft * 0.12);
  if (low.includes("?") || low.includes("!")) score = Math.min(1, score + 0.1);
  return r3(score);
}

const has = (text, ms) => ms.some((m) => text.toLowerCase().includes(m));
const userTurnsOf = (m) => m.filter((x) => x.role === "user").map((x) => x.content);

function firstWord(t) {
  return t.toLowerCase().trim().split(/\s+/)[0].replace(/[?!.,;:]/g, "");
}

function isEscalating(turns) {
  if (turns.length < 3) return false;
  return turns.slice(-3).every((t) => t.trim() && markers.QUESTION_OPENERS.includes(firstWord(t)));
}

function isDemand(turns) {
  const recent = turns.slice(-3).filter((t) => t.trim());
  if (recent.length < 3) return false;
  const short = recent.every((t) => t.trim().split(/\s+/).length <= 5);
  const punct = recent.filter((t) => "?!".includes(t.trim().slice(-1))).length;
  return short && punct >= 2;
}

function subtext(turns, annotated) {
  const current = turns.length ? turns[turns.length - 1] : "";
  if (has(current, markers.CORRECTION)) return "correction";
  if (has(current, markers.DISCLOSURE)) return "disclosure_request";
  if (["exhaustion", "resignation"].includes(annotated.subtext)) return annotated.subtext;
  if (isEscalating(turns)) return "escalating";
  if (isDemand(turns)) return "demand";
  const stripped = current.trimEnd();
  if (stripped.endsWith("?") && !has(current, markers.ESCALATION)) return "question";
  if (["!", "?"].includes(stripped.slice(-1)) || has(current, markers.ESCALATION)) return "challenge";
  return "statement";
}

function buildState(messages, annotated) {
  const turns = userTurnsOf(messages);
  const history = turns.map((text, i) => {
    const prev = i ? arousal(turns[i - 1]) : 0.0;
    return { arousal: arousal(text), escalation_delta: r3(arousal(text) - prev) };
  });
  const current = turns.length ? turns[turns.length - 1] : "";
  const prevTurn = turns.length > 1 ? turns[turns.length - 2] : "";
  const a = arousal(current);
  const state = {
    arousal: a,
    escalation_delta: prevTurn ? r3(a - arousal(prevTurn)) : 0.0,
    stance: annotated.stance || "unknown",
    subtext: subtext(turns, annotated),
    history,
  };
  state.escalating = state.escalation_delta > 0.15;
  return state;
}

function predicate(name, s) {
  const pastN = () => {
    const recent = s.history.slice(-4, -1);
    return recent.length >= 3 && recent.every((x) => x.escalation_delta > 0.05);
  };
  if (name === "any") return true;
  if (name === "escalating") return s.escalating;
  if (name === "escalating_past_n") return pastN();
  if (name === "not_escalating") return !s.escalating;
  if (name === "not_escalating_past_n") return !pastN();
  if (name === "user_is_right") return s.stance === "user_right";
  if (name === "user_is_wrong") return s.stance === "user_wrong";
  if (name === "stance_unknown") return s.stance === "unknown";
  if (name === "low_arousal") return s.arousal < 0.4;
  if (name === "high_arousal") return s.arousal >= 0.5;
  if (name === "high_arousal_run") {
    return s.history.length >= 2 && s.history.slice(-2).every((x) => x.arousal >= 0.5);
  }
  if (name === "subtext_set:drained") return ["exhaustion", "resignation"].includes(s.subtext);
  if (name.startsWith("subtext:")) {
    const t = name.slice("subtext:".length);
    if (!SUBTEXTS.includes(t)) throw new Error(`subtext not in SUBTEXTS: ${t}`);
    return s.subtext === t;
  }
  throw new Error("unknown predicate " + name);
}

function select(messages, annotated) {
  const state = buildState(messages, annotated);
  const applicable = POLICY_NAMES.filter((n) => policies[n].every((p) => predicate(p, state)));
  if (!applicable.length) return { name: "mirror_specific(default)", state, applicable };
  const best = Math.max(...applicable.map((n) => policies[n].length));
  return { name: applicable.find((n) => policies[n].length === best), state, applicable };
}

// --- run -----------------------------------------------------------------

const cases = py("eval/cases.jsonl").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));

let pass = 0;
const mismatches = [];
const seen = new Set();

for (const c of cases) {
  const sel = select(c.transcript, c.annotated || {});
  seen.add(sel.name);
  const ok = sel.name === c.expected_policy;
  if (ok) pass++;
  else mismatches.push([c.id, c.expected_policy, sel.name, sel.applicable]);
  console.log(
    `${c.id.padEnd(10)} subtext=${sel.state.subtext.padEnd(18)} ` +
    `exp=${c.expected_policy.padEnd(26)} got=${sel.name.padEnd(26)} ${ok ? "ok" : "MISMATCH"}`
  );
}

const unselected = POLICY_NAMES.filter((n) => !seen.has(n));

console.log(`\npolicy selection: ${pass}/${cases.length}`);
console.log(`unreachable policies: ${unselected.length ? unselected.join(", ") : "none"}`);
for (const [id, exp, got, app] of mismatches) {
  console.log(`  ${id}: expected ${exp}, got ${got} (applicable: ${app.join(", ") || "none"})`);
}

process.exit(mismatches.length || unselected.length ? 1 : 0);