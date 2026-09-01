// How much of the multimodality figure is sampling noise?
//
// The README's central claim is that the demonstrations have two lateral modes
// that average to roughly nothing, and it states one number per mode measured
// on one 800 demonstration set per seed. Three seeds is not enough to say how
// much of the spread between them is noise, and nothing in the repository had
// asked.
//
// This reruns the demonstrator from scratch. No torch, no shared code, its own
// xorshift generator and its own Box-Muller normals: 2000 independent replicates
// of the same 800 demonstration protocol, which is 38.4 million simulated steps
// and far more than the Python run could afford. That gives the sampling
// distribution of each published statistic, and every value in
// verify/demo_stats.json has to land inside its central 99 percent.
//
//   cd verify/demomc && cargo run --release -- ../demo_stats.json

use std::env;
use std::fs;

const OBSTACLE_Y: f64 = 0.0;
const OBSTACLE_HALF_W: f64 = 0.30;
const OBSTACLE_HALF_H: f64 = 0.10;
const MAX_SPEED: f64 = 0.16;
const SUCCESS_RADIUS: f64 = 0.16;
const DETOUR_GAP: f64 = 0.22;
const DETOUR_LIFT: f64 = 0.18;
const BELOW_MARGIN: f64 = 0.05;
const NOISE: f64 = 0.04;
const HORIZON: usize = 24;
const START_Y: f64 = -0.78;
const OBJECT_X: [f64; 3] = [-0.62, 0.0, 0.62];
const OBJECT_Y: f64 = 0.72;

const DEMOS: usize = 800;
const REPLICATES: usize = 2000;

struct Rng {
    s: u64,
    spare: Option<f64>,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { s: seed | 1, spare: None }
    }
    fn next_u64(&mut self) -> u64 {
        // xorshift64*, written out rather than pulled in
        let mut x = self.s;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.s = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f64 {
        // (0, 1)
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }
    fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let t = std::f64::consts::TAU * u2;
        self.spare = Some(r * t.sin());
        r * t.cos()
    }
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

#[derive(Default, Clone, Copy)]
struct Replicate {
    left_mean: f64,
    right_mean: f64,
    pooled_mean: f64,
    success: f64,
}

fn run_replicate(rng: &mut Rng) -> Replicate {
    let (mut ls, mut ln) = (0.0f64, 0usize);
    let (mut rs, mut rn) = (0.0f64, 0usize);
    let mut hits = 0usize;
    for _ in 0..DEMOS {
        let mut x = (rng.uniform() * 2.0 - 1.0) * 0.25;
        let mut y = START_Y;
        let tx = OBJECT_X[(rng.uniform() * 3.0) as usize % 3];
        let side: f64 = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
        for _ in 0..HORIZON {
            let below = y < OBSTACLE_Y + OBSTACLE_HALF_H + BELOW_MARGIN;
            let detour = side * (OBSTACLE_HALF_W + DETOUR_GAP);
            let (wx, wy) = if below {
                (detour, OBSTACLE_Y + OBSTACLE_HALF_H + DETOUR_LIFT)
            } else {
                (tx, OBJECT_Y)
            };
            let (dx, dy) = (wx - x, wy - y);
            let norm = (dx * dx + dy * dy).sqrt().max(1e-6);
            let ax = clamp(dx / norm + NOISE * rng.normal(), -1.0, 1.0);
            let ay = clamp(dy / norm + NOISE * rng.normal(), -1.0, 1.0);
            if below {
                if side < 0.0 {
                    ls += ax;
                    ln += 1;
                } else {
                    rs += ax;
                    rn += 1;
                }
            }
            let nx = clamp(x + ax * MAX_SPEED, -1.0, 1.0);
            let ny = clamp(y + ay * MAX_SPEED, -1.0, 1.0);
            let blocked = nx.abs() < OBSTACLE_HALF_W && (ny - OBSTACLE_Y).abs() < OBSTACLE_HALF_H;
            if !blocked {
                x = nx;
                y = ny;
            }
        }
        let d = ((x - tx).powi(2) + (y - OBJECT_Y).powi(2)).sqrt();
        if d < SUCCESS_RADIUS {
            hits += 1;
        }
    }
    Replicate {
        left_mean: ls / ln as f64,
        right_mean: rs / rn as f64,
        pooled_mean: (ls + rs) / (ln + rn) as f64,
        success: hits as f64 / DEMOS as f64,
    }
}

/// Every number that follows `"<key>":` in the file, in order.
fn json_numbers(text: &str, key: &str) -> Vec<f64> {
    let pat = format!("\"{}\":", key);
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(&pat) {
        rest = &rest[i + pat.len()..];
        let num: String = rest
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '.' || *c == 'e' || *c == '+')
            .collect();
        if let Ok(v) = num.parse::<f64>() {
            out.push(v);
        }
    }
    out
}

fn summarise(v: &mut Vec<f64>) -> (f64, f64, f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pick = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    (mean, sd, pick(0.005), pick(0.995))
}

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| "../demo_stats.json".to_string());
    let text = match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {}: {}", path, e);
            std::process::exit(2);
        }
    };

    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let mut reps = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        reps.push(run_replicate(&mut rng));
    }

    let fields: [(&str, fn(&Replicate) -> f64); 4] = [
        ("left_mean", |r| r.left_mean),
        ("right_mean", |r| r.right_mean),
        ("pooled_mean", |r| r.pooled_mean),
        ("scripted_success", |r| r.success),
    ];

    let mut failures = 0;
    let mut checked = 0;
    println!(
        "demomc: {} replicates of {} demonstrations, {} simulated steps",
        REPLICATES,
        DEMOS,
        REPLICATES * DEMOS * HORIZON
    );
    for (key, get) in fields {
        let mut vals: Vec<f64> = reps.iter().map(get).collect();
        let (mean, sd, lo, hi) = summarise(&mut vals);
        let published = json_numbers(&text, key);
        if published.is_empty() {
            eprintln!("  {}: no values found in {}", key, path);
            failures += 1;
            continue;
        }
        let mut worst = 0.0f64;
        for (seed, p) in published.iter().enumerate() {
            checked += 1;
            let z = (p - mean) / sd;
            if z.abs() > worst {
                worst = z.abs();
            }
            if *p < lo || *p > hi {
                println!(
                    "  FAIL {} seed {}: published {:.6} outside the central 99% [{:.6}, {:.6}]",
                    key, seed, p, lo, hi
                );
                failures += 1;
            }
        }
        println!(
            "  {:16} reference {:+.5} sd {:.5}, 99% [{:+.5}, {:+.5}], worst published seed {:.2} sd out",
            key, mean, sd, lo, hi, worst
        );
    }
    println!("demomc: {} published values checked, {} failures", checked, failures);
    if failures > 0 {
        std::process::exit(1);
    }
}
