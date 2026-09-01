# Logbook

## 2026-08-27, the encoder was the bottleneck, not the action head
**Tried:** first runs. Every head scored between 0.17 and 0.25 success against a scripted ceiling of 0.99, and the heads barely differed. Suspected the heads, spent three tuning passes there.
**Measured:** adding the agent's own position as an input lifted flow from 0.176 to 0.242 and regression from 0.191 to 0.219.
**Concluded:** the agent is one pixel at 24x24 and the encoder is three stride-2 convolutions into a global average pool, which destroys it. Every head was guessing where it was, so the comparison was measuring the encoder. A real robot has joint encoders, so proprioception is what a VLA actually receives and leaving it out was my error rather than a simplification. Check what the observation can physically represent before tuning anything downstream of it.

## 2026-08-27, action chunking made it worse, which I did not expect
**Tried:** the side the demonstrator takes is unobservable, so a head sampling independently each step can draw left then right and stall. Chunking should make the sampled mode persist, which is why ACT and pi-0 do it. Implemented it across all four heads and swept chunk 1 against 6.
**Measured:** flow went from 0.246 to 0.047 success and collisions rose from 6.09 to 12.57. Regression from 0.195 to 0.023.
**Concluded:** wrong, and clearly. Executing six actions open loop compounds prediction error faster than mode commitment buys anything on a task this reactive and this short. The reasoning was sound and the measurement disagreed, which is the useful kind of negative result. Kept the machinery behind a flag and recorded the number rather than deleting the experiment. Would expect the opposite on a longer horizon task with more accurate per step predictions, which is presumably where ACT lives.

## 2026-08-27, the held out split leaked and a test caught it
**Tried:** wrote a test asserting the evaluation split contains only held out (colour, shape) pairs.
**Measured:** it failed. With two held out pairs, evaluation scenes contained (0, 0), which is a training pair.
**Concluded:** the scene sampler used randperm(len(allowed))[:3], which silently returns fewer than three indices when the split is small, leaving the remaining object slots at their zero initialised value, that is a red square. So generalisation evaluation contained a training object with no error raised anywhere. It now raises with a message naming the split size. The experiment holds out exactly three pairs so the real runs were unaffected, but nothing was stopping that from changing.

## 2026-08-27, mode averaging, measured, and it separates cleanly
**Tried:** all four heads, 3 seeds, 800 demonstrations, 2000 steps each. 1042 s.
**Measured:** regression spends 8.44 of 24 steps pressed into the obstacle; discrete bins 2.05, diffusion 2.39, flow 3.56. The seed ranges do not overlap: regression 8.20 to 9.45 against a multimodal envelope of 1.43 to 4.37. Success also separates for flow alone, 0.270 to 0.297 against regression's 0.211 to 0.223. On latency, regression runs at 1.89M Hz and diffusion at 50 steps at 20.9k, a factor of 90. Flow at a single step reaches 0.309 success at 1.17M Hz.
**Concluded:** the theory predicted this exactly and it reproduced. The demonstrations average to -0.004 lateral action, which points at the wall, and the head that minimises squared error drives there. The pi-0 claim also holds on this task: flow keeps multimodality at within 1.6x of regression's control rate while diffusion pays 90x. Worth being careful about what this does not show: every head is far below the 0.988 demonstrator, so avoiding the obstacle is necessary and not sufficient, and none of this tests the pretraining claim that actually motivates VLAs.
