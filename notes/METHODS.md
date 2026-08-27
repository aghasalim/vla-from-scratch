# Methods and detail

Long form detail moved out of the README.


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
