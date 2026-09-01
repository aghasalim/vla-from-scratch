// The figures that are ratios of other figures, and the demonstration
// statistic, checked in Node.
//
// scripts/check_numbers.py greps the documents for each median it recomputes.
// That leaves two holes. The first is every number written as a multiple: "4.0x
// slower", "1.9x the next one", "within 1.6x of regression's". Those are
// arithmetic on the published rates and nothing recomputed them. The second is
// the demonstration statistic itself, which lives in no results CSV at all, so
// the drift check cannot see it.
//
// This recomputes both, from results/heads.csv, results/step-sweep.csv and
// verify/demo_stats.json, with its own CSV parser and its own median, and
// requires the documents to say what the arithmetic says.
//
//   node verify/derived.js

const fs = require("fs");
const path = require("path");

const root = process.argv[2] || ".";
const read = (p) => fs.readFileSync(path.join(root, p), "utf8");

function parseCSV(text) {
  // the results files hold plain numbers and head names with no commas or
  // quotes in them, so a split is honest here rather than a shortcut
  const lines = text.trim().split("\n").map((l) => l.replace(/\r$/, ""));
  const cols = lines[0].split(",");
  return lines.slice(1).map((l) => {
    const cells = l.split(",");
    if (cells.length !== cols.length) {
      throw new Error(`ragged row: ${cells.length} cells, header has ${cols.length}`);
    }
    return Object.fromEntries(cols.map((c, i) => [c, cells[i]]));
  });
}

const median = (v) => {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

const heads = parseCSV(read("results/heads.csv"));
const sweep = parseCSV(read("results/step-sweep.csv"));
const demo = JSON.parse(read("verify/demo_stats.json"));

const hz = (name) =>
  median(heads.filter((r) => r.head === name).map((r) => +r.max_hz));
const sweepHz = (name, steps) =>
  median(sweep.filter((r) => r.head === name && +r.steps === steps).map((r) => +r.max_hz));

const docs = (read("README.md") + "\n" + read("notes/METHODS.md")).replace(/\s+/g, " ");

let checked = 0;
const failures = [];
function must(label, re, expected) {
  checked++;
  const m = docs.match(re);
  if (!m) {
    failures.push(`${label}: the sentence this reads is no longer in the documents`);
  } else if (m.slice(1).join(" ") !== expected) {
    failures.push(`${label}: documents say "${m.slice(1).join(" ")}", recomputed ${expected}`);
  }
}

// the "vs regression" column of the latency table, both copies of it
const reg = hz("regression");
const slower = (name, digits) => (reg / hz(name)).toFixed(digits);
must("discrete bins vs regression", /discrete bins \| 2 \| [\d,]+ \| ([\d.]+)x slower \|/,
     slower("discrete bins", 1));
must("flow vs regression", /flow, 5 steps \| 5 \| [\d,]+ \| ([\d.]+)x slower \|/,
     slower("flow (pi-0 style)", 1));
must("diffusion vs regression", /diffusion, 50 steps \| 50 \| [\d,]+ \| \*\*([\d.]+)x slower\*\* \|/,
     slower("diffusion", 0));

// the three multiples in the step sweep paragraph
must("flow at 1 step against flow at 2", /([\d.]+)x the next one/,
     (sweepHz("flow (pi-0 style)", 1) / sweepHz("flow (pi-0 style)", 2)).toFixed(1));
must("flow at 1 step against flow at 5", /([\d.]+)x the flow 5 step row/,
     (sweepHz("flow (pi-0 style)", 1) / sweepHz("flow (pi-0 style)", 5)).toFixed(1));
must("flow at 1 step against regression", /within ([\d.]+)x of regression's/,
     (reg / sweepHz("flow (pi-0 style)", 1)).toFixed(1));

// the demonstration statistic, which no results CSV holds
const per = Object.values(demo.per_seed);
const stat = (k) => median(per.map((s) => s[k]));
must("demonstration modes",
     /left mode's lateral action averages (-[\d.]+), the right mode's \+([\d.]+), and the two together average (-?[\d.]+)/,
     [stat("left_mean").toFixed(2), stat("right_mean").toFixed(2),
      stat("pooled_mean").toFixed(3)].join(" "));
must("scripted demonstrator success", /scripted demonstrator's ([\d.]+)/,
     stat("scripted_success").toFixed(3));
must("scripted demonstrator success, methods copy", /scripted demonstrator scores ([\d.]+)/,
     stat("scripted_success").toFixed(3));

console.log(`derived.js: ${checked} derived figures recomputed, ${failures.length} failures`);
for (const f of failures) console.log("  -", f);
process.exit(failures.length ? 1 : 0);
