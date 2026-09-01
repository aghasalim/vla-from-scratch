// Structural validation of every committed results file, plus an independent
// recomputation of the summary figures the README and notes/METHODS.md quote.
//
// results/heads.csv and results/step-sweep.csv hold one row per (head, seed).
// Every figure in the two documents is a median, a minimum or a maximum over
// those rows, computed once by pandas and once by scripts/check_numbers.py,
// both in Python and both reading the same code path. This recomputes them in
// Go, with its own CSV reader and its own median, and requires the documents to
// state what the files actually say.
//
//	cd verify/gocheck && go run . -root ..
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const horizon = 24 // steps per episode, vla/envs.py ReachEnv default

type table struct {
	name string
	cols []string
	rows [][]string
}

func (t *table) col(name string) int {
	for i, c := range t.cols {
		if c == name {
			return i
		}
	}
	return -1
}

func (t *table) num(row int, name string) (float64, error) {
	i := t.col(name)
	if i < 0 {
		return 0, fmt.Errorf("%s: no column %q", t.name, name)
	}
	return strconv.ParseFloat(t.rows[row][i], 64)
}

var fails []string

func fail(format string, a ...any) {
	fails = append(fails, fmt.Sprintf(format, a...))
}

// readCSV also does the structural validation: a ragged row, a duplicated
// column name or a non-finite number is rejected here rather than quietly
// becoming a wrong average later.
func readCSV(path string) (*table, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	r := csv.NewReader(fh)
	r.FieldsPerRecord = 0 // enforce a constant field count: ragged rows error
	recs, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	name := filepath.Base(path)
	if len(recs) < 2 {
		return nil, fmt.Errorf("%s: header plus %d data rows", name, len(recs)-1)
	}
	t := &table{name: name, cols: recs[0], rows: recs[1:]}
	seen := map[string]bool{}
	for _, c := range t.cols {
		if strings.TrimSpace(c) == "" {
			return nil, fmt.Errorf("%s: empty column name", name)
		}
		if seen[c] {
			return nil, fmt.Errorf("%s: duplicate column %q", name, c)
		}
		seen[c] = true
	}
	for i, row := range t.rows {
		for j, cell := range row {
			s := strings.TrimSpace(cell)
			if s == "" {
				return nil, fmt.Errorf("%s: empty cell at row %d column %q", name, i+2, t.cols[j])
			}
			if v, err := strconv.ParseFloat(s, 64); err == nil {
				if math.IsNaN(v) || math.IsInf(v, 0) {
					return nil, fmt.Errorf("%s: non-finite %q at row %d column %q",
						name, cell, i+2, t.cols[j])
				}
			} else if strings.EqualFold(s, "nan") || strings.EqualFold(s, "inf") ||
				strings.EqualFold(s, "-inf") {
				return nil, fmt.Errorf("%s: non-finite %q at row %d column %q",
					name, cell, i+2, t.cols[j])
			}
		}
	}
	return t, nil
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func minMax(v []float64) (float64, float64) {
	lo, hi := v[0], v[0]
	for _, x := range v {
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	return lo, hi
}

// thousands formats like the documents do: 1891644 -> "1,891,644".
func thousands(v float64) string {
	s := strconv.FormatFloat(math.Round(v), 'f', 0, 64)
	neg := strings.HasPrefix(s, "-")
	s = strings.TrimPrefix(s, "-")
	var out []string
	for len(s) > 3 {
		out = append([]string{s[len(s)-3:]}, out...)
		s = s[:len(s)-3]
	}
	out = append([]string{s}, out...)
	r := strings.Join(out, ",")
	if neg {
		r = "-" + r
	}
	return r
}

var claims int

// stated requires the exact figure to appear in the prose, not merely as a
// substring of a longer number.
func stated(text, figure, label string) {
	claims++
	re := regexp.MustCompile(`(?:^|[^\d.,])` + regexp.QuoteMeta(figure) + `(?:[^\d]|$)`)
	if !re.MatchString(text) {
		fail("%s should read %s, not found in the documents", label, figure)
	}
}

// statedEither accepts either spelling of an integer, since the documents write
// 2000 gradient steps without a separator and 19,200 transitions with one.
func statedEither(text string, v int, label string) {
	claims++
	plain, grouped := strconv.Itoa(v), thousands(float64(v))
	for _, f := range []string{plain, grouped} {
		re := regexp.MustCompile(`(?:^|[^\d.,])` + regexp.QuoteMeta(f) + `(?:[^\d]|$)`)
		if re.MatchString(text) {
			return
		}
	}
	fail("%s should read %s or %s, not found in the documents", label, plain, grouped)
}

func main() {
	root := flag.String("root", "..", "the verify/ directory; the repository is its parent")
	flag.Parse()
	repo := filepath.Join(*root, "..")
	res := func(p string) string { return filepath.Join(repo, "results", p) }

	files := []string{"heads.csv", "latency.csv", "step-sweep.csv", "success.csv"}
	loaded := map[string]*table{}
	for _, f := range files {
		t, err := readCSV(res(f))
		if err != nil {
			fail("%v", err)
			continue
		}
		loaded[f] = t
	}

	var meta struct {
		Seeds  []int   `json:"seeds"`
		Demos  int     `json:"demos"`
		Steps  int     `json:"steps"`
		EvalN  int     `json:"eval_n"`
		Wall   float64 `json:"wall_clock_s"`
		Device string  `json:"device"`
	}
	raw, err := os.ReadFile(res("run-meta.json"))
	if err != nil {
		fail("run-meta.json: %v", err)
	} else if err := json.Unmarshal(raw, &meta); err != nil {
		fail("run-meta.json: %v", err)
	}

	readDoc := func(p string) string {
		b, err := os.ReadFile(filepath.Join(repo, p))
		if err != nil {
			fail("%s: %v", p, err)
			return ""
		}
		return string(b)
	}
	docs := readDoc("README.md") + "\n" + readDoc("notes/METHODS.md")

	// The run the documents describe has to be the run the files came from.
	// Running the sweep with the argparse defaults instead of the documented
	// arguments produces a different heads.csv, and nothing else notices.
	stated(docs, strconv.Itoa(len(meta.Seeds)), "seed count from run-meta.json")
	stated(docs, thousands(float64(meta.Demos)), "demos from run-meta.json")
	statedEither(docs, meta.Steps, "gradient steps from run-meta.json")
	stated(docs, thousands(float64(meta.EvalN)), "eval episodes from run-meta.json")
	stated(docs, thousands(float64(meta.Demos*horizon)), "demo transitions, demos x horizon")
	stated(docs, strconv.Itoa(int(meta.Wall/60)), "wall clock minutes from run-meta.json")

	if h := loaded["heads.csv"]; h != nil {
		by := map[string][]int{}
		var order []string
		for i, r := range h.rows {
			k := r[h.col("head")]
			if _, ok := by[k]; !ok {
				order = append(order, k)
			}
			by[k] = append(by[k], i)
		}
		if want := len(meta.Seeds) * len(order); len(h.rows) != want {
			fail("heads.csv has %d rows, expected %d heads x %d seeds",
				len(h.rows), len(order), len(meta.Seeds))
		}
		var allMM []float64
		var regBlocked []float64
		for _, head := range order {
			get := func(col string) []float64 {
				var out []float64
				for _, i := range by[head] {
					v, err := h.num(i, col)
					if err != nil {
						fail("%v", err)
						return nil
					}
					out = append(out, v)
				}
				return out
			}
			succ, unseen, blocked, hz := get("success"), get("unseen_success"),
				get("blocked_steps"), get("max_hz")
			if succ == nil || unseen == nil || blocked == nil || hz == nil {
				continue
			}
			stated(docs, fmt.Sprintf("%.3f", median(succ)), head+" median success")
			stated(docs, fmt.Sprintf("%.3f", median(unseen)), head+" median unseen success")
			stated(docs, fmt.Sprintf("%.2f", median(blocked)), head+" median collisions")
			stated(docs, thousands(median(hz)), head+" median max Hz")
			lo, hi := minMax(succ)
			stated(docs, fmt.Sprintf("%.3f", lo), head+" success min")
			stated(docs, fmt.Sprintf("%.3f", hi), head+" success max")
			if head == "regression" {
				regBlocked = blocked
			} else {
				allMM = append(allMM, blocked...)
			}
			// max_hz is 1/latency_s by construction; drift here means one of the
			// two columns was edited without the other.
			for _, i := range by[head] {
				lat, _ := h.num(i, "latency_s")
				got, _ := h.num(i, "max_hz")
				if rel := math.Abs(1/lat-got) / got; rel > 1e-9 {
					fail("heads.csv row %d: max_hz %.6g is not 1/latency_s (%.6g)",
						i+2, got, 1/lat)
				}
			}
		}
		if regBlocked != nil {
			lo, hi := minMax(regBlocked)
			stated(docs, fmt.Sprintf("%.2f", lo), "regression collision min")
			stated(docs, fmt.Sprintf("%.2f", hi), "regression collision max")
		}
		if allMM != nil {
			lo, hi := minMax(allMM)
			stated(docs, fmt.Sprintf("%.2f", lo), "multimodal collision envelope min")
			stated(docs, fmt.Sprintf("%.2f", hi), "multimodal collision envelope max")
		}
	}

	if s := loaded["step-sweep.csv"]; s != nil {
		type key struct {
			head  string
			steps string
		}
		by := map[key][]int{}
		var order []key
		for i, r := range s.rows {
			k := key{r[s.col("head")], r[s.col("steps")]}
			if _, ok := by[k]; !ok {
				order = append(order, k)
			}
			by[k] = append(by[k], i)
		}
		if want := len(meta.Seeds) * len(order); len(s.rows) != want {
			fail("step-sweep.csv has %d rows, expected %d configurations x %d seeds",
				len(s.rows), len(order), len(meta.Seeds))
		}
		for _, k := range order {
			var succ, hz []float64
			for _, i := range by[k] {
				a, _ := s.num(i, "success")
				b, _ := s.num(i, "max_hz")
				succ = append(succ, a)
				hz = append(hz, b)
			}
			label := fmt.Sprintf("%s at %s steps", k.head, k.steps)
			stated(docs, fmt.Sprintf("%.3f", median(succ)), label+" median success")
			stated(docs, thousands(median(hz)), label+" median max Hz")
		}
	}

	fmt.Printf("gocheck: %d results files structurally valid, %d quoted figures recomputed, %d failures\n",
		len(loaded)+1, claims, len(fails))
	for i, f := range fails {
		if i == 12 {
			fmt.Printf("  ... and %d more\n", len(fails)-12)
			break
		}
		fmt.Println("  -", f)
	}
	if len(fails) > 0 {
		os.Exit(1)
	}
}
