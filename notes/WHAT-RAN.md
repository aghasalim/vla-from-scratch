# What ran

The tables come from a single run of `experiments/main.py`. Every setting of
that run is either on the command line, in the code at a line named below, or
in a column of the files it wrote. Where a value is in none of those places the
line says so instead.

## Command

```
python -m experiments.main --seeds 0 1 2 --demos 800 --steps 2000
```

From the README, "How to run it". `results/run-meta.json` holds `demos` 800
and `steps` 2000, written from the parsed arguments at
`experiments/main.py:112`, so the file and the README agree. The defaults at
`experiments/main.py:62` and `:63` are 600 and 2500, which is not what ran.

## Date, machine, versions

| | | source |
|---|---|---|
| date | 2026-08-27 | commit `02e276a` brought in `heads.csv` and `step-sweep.csv`; the logbook entry describing the run is dated the same day |
| CPU | Apple M4 | README, "about 17 minutes on an M4 CPU" |
| device | cpu | `results/run-meta.json` |
| torch | 2.13.0 | `results/run-meta.json` |

## Time

`wall_clock_s` in `run-meta.json` is 1041.97 s, measured from
`experiments/main.py:72` to `:114`. Of that, the twelve training loops account
for 999.7 s: `train_s` in `results/heads.csv`, summed. The per head figures sit
between 82.5 s and 84.8 s and barely depend on the head, because the encoder at
`experiments/main.py:38` dominates and is the same for all four. The other 42 s
is demonstration collection, 42 rollouts and 30 latency calls, none of which
has a timing column.

## Seeds

0, 1 and 2, the `seed` column. What each one controls:

| | seed | line |
|---|---|---|
| demonstrations | seed | `experiments/main.py:75` |
| encoder and head init | seed | `experiments/main.py:37` |
| batch indices | seed + 3 | `experiments/main.py:45` |
| held in rollouts | seed + 500 | `experiments/main.py:81`, `:99` |
| held out rollouts | seed + 900 | `experiments/main.py:83` |
| latency scenes | 0 | `vla/eval.py:50`, the same 64 scenes for every head and seed |

## Settings

| setting | value | in `run-meta.json` | read at |
|---|---:|:---:|---|
| demos | 800 | yes | `experiments/main.py:75` |
| steps | 2000 | yes | `:43`, `:47` |
| batch | 256 | yes | `:48` |
| lr | 1e-3 | yes | `:41`, `:42` |
| eval_n | 256 | yes | `:81`, `:83`, `:99` |
| chunk | 1 | yes | `:39`, `:76` |
| held out pairs | (0, 1), (1, 2), (2, 0) | yes | `experiments/main.py:32` |
| optimiser | AdamW | no | `experiments/main.py:41` |
| schedule | OneCycleLR, pct_start 0.1 | no | `experiments/main.py:42` to `:43` |
| gradient clip | 1.0 | no | `experiments/main.py:53` |
| episode horizon | 24 | no | `vla/eval.py:12` |
| sampling sweep | diffusion 50, 10, 4; flow 5, 2, 1 | no | `experiments/main.py:33` |
| latency batch, repeats | 64, 12 | no | `vla/eval.py:48` |

The 19,200 transitions the README mentions are 800 demonstrations of 24 steps.
The `final_loss` column is the loss of the last batch, `experiments/main.py:56`.
It is one number per run, not a series, which is why there is no loss curve in
`results/`: the loop at `experiments/main.py:47` records nothing along the way.
For the record the last batch losses span 0.050 to 0.058 for regression, 0.19 to
0.23 for diffusion, 0.28 to 0.33 for flow and 1.37 to 1.41 for discrete bins,
and the four are not on the same scale so they do not rank the heads.

## Outputs of that run

| file | rows | written at |
|---|---:|---|
| `results/heads.csv` | 12, one per (seed, head) | `experiments/main.py:105` |
| `results/step-sweep.csv` | 18, one per (seed, head, steps) | same loop |
| `results/run-meta.json` | | `experiments/main.py:112` |

`results/rollout-traces.npz` came later, in commit `382beb2` on 2026-08-28.
`bench/figures.py:198` to `:213` retrains regression and flow at seed 0, checks
the result against the seed 0 rows of `heads.csv` and only then saves the agent
positions. It holds paths, not parameters.

## Two files this script did not write

`results/success.csv` and `results/latency.csv` were committed alongside the
others but were produced by an earlier harness that is no longer in the tree.
They have been removed. The logbook entry for 2026-09-13 says what they were
and why they went.
