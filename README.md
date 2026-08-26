# vla-from-scratch

A vision-language-action policy: a pretrained VLM backbone fine-tuned to output robot actions, with a controlled comparison of the four action-representation choices that the field is currently split over.

> **Status: scaffold. Nothing here is built or measured yet.**
> This repo currently holds the project specification, the shared agent conventions,
> and an empty logbook. Every number in the tables below is a `TODO` because no
> experiment has been run. The `prompts/` task specs referenced in the wave table
> are not written yet either.
>
> Nothing in this repo is estimated or taken from a paper. When a table has a number
> in it, that number came from a run in `results/`.

---

## Why

The bet behind VLAs is that a model pretrained on web-scale vision-language data brings *semantic* generalization to robotics — recognizing an object it never saw in a demonstration, or following an instruction phrased in a new way. That's a real and testable claim.

The open engineering question is how a language model should emit continuous actions at 50Hz. Four answers are in active use: discrete bins as text tokens (RT-2), a continuous regression head, a diffusion policy head, and a flow-matching action expert (π0). **Task 03 compares all four under identical conditions**, which is the part of this repo that isn't a reimplementation of anything.

If repo 04 is built, its flow-matching machinery is exactly what the π0-style head needs.

## Scope

**Simulation only. No robot required.** LIBERO or ManiSkill. Real-robot evaluation is a different project with a different budget, and sim results are a legitimate contribution as long as you say they're sim results.

## Hardware

- **GPU:** `TODO — python -m scripts.env`
- 16GB minimum, 24GB comfortable. A ~1–3B VLM with LoRA fits; a 7B full fine-tune does not.

## Results

LIBERO success rate (%), 3 seeds, 50 rollouts per task suite:

| Method | Spatial | Object | Goal | Long | Unseen objects | Novel phrasing |
|---|---:|---:|---:|---:|---:|---:|
| BC from scratch | TODO | TODO | TODO | TODO | TODO | TODO |
| ACT (chunked) | TODO | TODO | TODO | TODO | TODO | TODO |
| VLA + discrete bins | TODO | TODO | TODO | TODO | TODO | TODO |
| VLA + regression | TODO | TODO | TODO | TODO | TODO | TODO |
| VLA + diffusion head | TODO | TODO | TODO | TODO | TODO | TODO |
| VLA + flow head (π0) | TODO | TODO | TODO | TODO | TODO | TODO |

The last two columns are the ones that justify the VLM. If a from-scratch BC policy matches the VLA there, the pretraining bought nothing.

Control-rate feasibility:

| Action head | Inference latency | Max control Hz | Chunk size |
|---|---:|---:|---:|

## Waves

```
00 bootstrap + sim + eval                (serial)
   ├─ 01 BC and ACT baselines            ┐
   └─ 02 VLM backbone + features         ┘ parallel
        └─ 03 action representation study (serial — the core)
             ├─ 04 chunking + latency     ┐
             └─ 05 generalization eval    ┘ parallel
                  └─ 06 writeup
```

| Task | OWNS | READS |
|---|---|---|
| 00 | `scripts/`, `Makefile`, `vla/envs/`, `vla/data/`, `vla/eval/` | — |
| 01 | `vla/baselines/`, `train/train_bc.py` | `vla/data/`, `vla/eval/` |
| 02 | `vla/backbone/` | `vla/data/` |
| 03 | `vla/heads/`, `train/train_vla.py` | `vla/backbone/`, `vla/baselines/` |
| 04 | `vla/chunking.py`, `bench/latency.py` | `vla/heads/` |
| 05 | `experiments/generalization/` | `vla/heads/`, `vla/eval/` |
| 06 | `bench/`, `notes/paper.md`, `README.md` | everything |

See [`CONVENTIONS.md`](CONVENTIONS.md).

## Author

Aghasalim Mustafazada — third-year AI student at Howest, Belgium.

<p align="center">
  <a href="https://github.com/aghasalim">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="github"></a>
  <a href="https://www.kaggle.com/aghasalimmustafazada">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="kaggle"></a>
  <a href="https://linkedin.com/in/mustafazada">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"></a>
  <a href="https://orcid.org/0009-0001-8746-4582">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="orcid"></a>
</p>

## License

MIT — see [LICENSE](LICENSE).
