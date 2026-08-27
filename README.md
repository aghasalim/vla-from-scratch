# vla-from-scratch

Four ways for a policy to emit continuous actions, compared under identical
conditions: discrete bins as tokens (RT-2), a regression head, a diffusion
policy head, and a flow matching action expert (pi-0).

That comparison is the part of this project that is not a reimplementation of
anything, and it is the part that runs on a laptop CPU. What does not run here
is the thing that makes a VLA a VLA, and section two says so before showing any
numbers.

![mode averaging](results/multimodality.png)

## Scope, stated first

A real VLA puts a pretrained 1B to 3B vision language model in front of the
action head, and the claim being tested is that web scale pretraining brings
*semantic* generalisation to robotics. **This repo does not test that claim and
cannot.** The encoder here is a small CNN plus token embeddings, trained from
scratch on 19,200 transitions of a 2D task. Nothing is pretrained, so there is
no transfer to measure.

There is also no LIBERO and no ManiSkill. The simulator is 24x24 pixels of a 2D
reaching task, written directly so the repo has no simulator dependency.

What survives that reduction is the engineering question the four heads answer:
given identical perception, how should a policy represent a continuous action?
Every head below sits on the same features, trained with the same optimiser for
the same number of steps, so differences between them are about the action
representation and nothing else.

## The task, and why it has an obstacle

An agent at the bottom, three objects at the top, and an instruction naming one
of them by colour and shape. A wall sits in the middle.

The wall is the entire design. A demonstrator can route around it to the left or
the right, and both are correct, so the demonstration distribution is
**multimodal**. Measured on the demonstrations: the left mode's lateral action
averages -0.42, the right mode's +0.42, and the two together average +0.02,
which points straight at the wall.

That last number is the whole experiment. A regression head trained with mean
squared error converges to the conditional mean, so on this data it learns to
drive into the obstacle. Without multimodality a regression head would win and
the comparison would say nothing at all.

## Results

3 seeds, 800 demonstrations each, 2000 gradient steps per head, 17 minutes on a
CPU. The scripted demonstrator scores 0.988, which is the ceiling.

| head | success | range | held out pairs | obstacle collisions | NFE | max Hz |
|---|---:|---|---:|---:|---:|---:|
| regression | 0.219 | 0.211 to 0.223 | 0.172 | **8.44** | 1 | 1,891,644 |
| discrete bins | 0.234 | 0.211 to 0.289 | 0.195 | 2.05 | 2 | 472,761 |
| diffusion | 0.238 | 0.184 to 0.254 | 0.188 | 2.39 | 50 | 20,913 |
| **flow (pi-0)** | **0.273** | 0.270 to 0.297 | **0.238** | 3.56 | 5 | 254,558 |

**The mode averaging prediction holds, and cleanly.** Regression spends 8.44
steps of a 24 step episode pressed into the wall. Every multimodal head spends
between 2.05 and 3.56. The seed ranges do not overlap at all: regression spans
8.20 to 9.45, and the widest multimodal range is 1.43 to 4.37. This is the
clearest result in the repo and it is exactly what the theory predicts.

**Success separates too, but only for flow.** Flow spans 0.270 to 0.297 against
regression's 0.211 to 0.223, which do not overlap. Discrete bins and diffusion
sit in between with ranges that overlap regression's, so they are not separated
by success even though they clearly are by collisions.

**Everything falls far short of the 0.988 demonstrator.** The best head reaches
0.273. Avoiding the obstacle is necessary and not sufficient: a policy also has
to arrive and stop within 0.16 of a target it can only locate from a 24x24
image, and none of these do that reliably. So the collision result should be
read as a clean demonstration of the mechanism, not as a claim that any of these
heads solves the task.

## Latency is where flow earns its place

![control rate against success](results/latency.png)

| head | NFE | max Hz | vs regression |
|---|---:|---:|---:|
| regression | 1 | 1,891,644 | 1.0x |
| discrete bins | 2 | 472,761 | 4.0x slower |
| flow, 5 steps | 5 | 254,558 | 7.4x slower |
| diffusion, 50 steps | 50 | 20,913 | **90x slower** |

Sweeping the sampling budget is the useful part:

| head | steps | success | max Hz |
|---|---:|---:|---:|
| diffusion | 50 | 0.238 | 20,985 |
| diffusion | 10 | 0.223 | 105,379 |
| diffusion | 4 | 0.250 | 258,237 |
| flow | 5 | 0.266 | 251,679 |
| flow | 2 | 0.273 | 616,867 |
| **flow** | **1** | **0.309** | **1,167,180** |

Flow at a single integration step is the best configuration measured here on
both axes: 0.309 success at 1.17M Hz, which is within 1.6x of the regression
head's rate while keeping the multimodality that regression lacks. That is the
pi-0 argument, and on this task it holds.

Diffusion at 4 steps is not worse than at 50, which says the 50 step default is
simply overpaying on a 2D action. On a higher dimensional action it would not
be, and I would not generalise this to a real robot.

Note these Hz numbers are the action head only, on 2D actions, and they exclude
the encoder forward pass. They compare heads against each other and are not a
claim about any real control loop.

## Generalisation

![held out pairs](results/generalisation.png)

Three (colour, shape) pairs are withheld from training entirely. Every head
drops: flow from 0.273 to 0.238, regression from 0.219 to 0.172. Nothing here
was pretrained, so this measures whether the compositional structure was learned
from 19,200 in-domain transitions, not whether pretraining transfers. The real
version of this column is the one that justifies putting a VLM in the loop, and
it needs the VLM.

## What I got wrong

**I predicted action chunking would fix the dithering, and it made things worse.**
The reasoning was sound: the demonstrator's choice of side is unobservable, so a
head that samples independently every step can draw left then right and stall.
Committing to a chunk should make the mode persist. Measured with chunk 6, flow
went from 0.246 to 0.047 success and collisions rose from 6.09 to 12.57. Open
loop execution compounds prediction error faster than mode commitment buys
anything on a task this reactive. The chunking machinery is still in the code
behind a flag, and the result is recorded rather than deleted.

**My held out split leaked, and a test caught it.** The scene sampler took
`randperm(len(allowed))[:3]`, which silently returns fewer than three indices
when the split has fewer than three pairs, leaving the remaining object slots at
their zero initialised value, that is a (red, square) that the split was meant to
exclude. Held out evaluation would have contained a training object with no error
raised. It now raises with a message naming the split.

**I built the encoder without proprioception and spent three passes tuning around
it.** The agent is one pixel at 24x24, and three stride-2 convolutions into a
global average pool destroy that, so every head was guessing where it was. A real
robot has joint encoders. Adding the agent position lifted flow from 0.176 to
0.242 and the comparison stopped measuring the encoder.

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

```bash
python -m experiments.main --seeds 0 1 2 --demos 800 --steps 2000
```

```bash
python -m bench.figures
```

The sweep takes about 17 minutes on an M4 CPU. Figures read the committed CSVs
and never re-run an experiment.

## Layout

```
vla/envs.py      the 2D task, written directly, no simulator dependency
vla/data.py      scripted demonstrations, deliberately bimodal
vla/backbone.py  small vision language encoder, shared by every head
vla/heads.py     the four action representations, all chunk aware
vla/eval.py      closed loop rollouts and latency
experiments/     the sweep
tests/           29 tests
```

## Sources

- **Brohan, Brown, Carbajal et al. RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. CoRL 2023.** [arXiv:2307.15818](https://arxiv.org/abs/2307.15818) The discrete bins as text tokens head.
- **Black, Brown, Darpinian et al. pi-0: A Vision-Language-Action Flow Model for General Robot Control. 2024.** [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) The flow matching action expert, and the latency argument this repo measures.
- **Chi, Feng, Du et al. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. RSS 2023.** [arXiv:2303.04137](https://arxiv.org/abs/2303.04137) The diffusion head, and the clearest statement of why multimodality in demonstrations breaks regression.
- **Zhao, Kumar, Levine, Finn. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. RSS 2023.** [arXiv:2304.13705](https://arxiv.org/abs/2304.13705) ACT and action chunking, the idea my negative result above tested.
- **Kim, Pertsch, Karamcheti et al. OpenVLA: An Open-Source Vision-Language-Action Model. CoRL 2024.** [arXiv:2406.09246](https://arxiv.org/abs/2406.09246) The open reference point for the full system.
- **Liu, Zhu, Gao et al. LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. NeurIPS 2023.** [arXiv:2306.03310](https://arxiv.org/abs/2306.03310) The benchmark this repo substitutes a toy for, and why that substitution costs the generalisation claim.

Related: [rectified-flow-from-scratch](https://github.com/aghasalim/rectified-flow-from-scratch)
builds the flow matching machinery the pi-0 style head uses here.

## Methodology

The rules this follows are in [`METHODOLOGY.md`](METHODOLOGY.md). Rule 15, say "not measured"
rather than extrapolating, is why the Scope section leads.

## Author

Aghasalim Mustafazada, third year AI student at Howest, Belgium.

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

MIT, see [LICENSE](LICENSE).
