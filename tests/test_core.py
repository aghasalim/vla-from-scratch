"""Tests. The load bearing ones are about multimodality and chunk shapes."""
import pytest
import torch

from vla.backbone import VisionLanguageEncoder
from vla.data import collect, waypoint_route
from vla.envs import OBSTACLE_HALF_W, SUCCESS_RADIUS, ReachEnv
from vla.eval import latency, rollout
from vla.heads import HEADS, DiscreteBins, RegressionHead


# --- environment ------------------------------------------------------------
def test_obstacle_blocks_the_straight_line():
    """If the straight path were free, the demonstrations would be unimodal and
    the whole comparison in this repo would be vacuous."""
    e = ReachEnv(2, seed=0)
    assert e.hits_obstacle(torch.tensor([[0.0, 0.0]]))[0]
    assert not e.hits_obstacle(torch.tensor([[0.9, 0.0]]))[0]


def test_agent_cannot_move_into_the_obstacle():
    e = ReachEnv(1, seed=0)
    e.agent = torch.tensor([[0.0, -0.2]])
    before = e.agent.clone()
    for _ in range(6):
        e.step(torch.tensor([[0.0, 1.0]]))
    assert e.agent[0, 1] < 0.0, "agent passed through the obstacle"
    assert e.agent[0, 1] >= before[0, 1]


def test_observation_contains_image_language_and_state():
    o = ReachEnv(4, seed=0).observe()
    assert o["image"].shape == (4, 3, 24, 24)
    assert o["color"].shape == (4,) and o["shape"].shape == (4,)
    assert o["state"].shape == (4, 2)


def test_held_out_pairs_are_actually_held_out():
    held = {(0, 1), (1, 2), (2, 0)}
    tr = ReachEnv(64, seed=0, held_out_pairs=held, train=True)
    seen = {(int(c), int(s)) for c, s in zip(tr.obj_color.flatten(), tr.obj_shape.flatten())}
    assert not (seen & held), f"training scenes contain held out pairs: {seen & held}"
    te = ReachEnv(64, seed=0, held_out_pairs=held, train=False)
    seen_t = {(int(c), int(s)) for c, s in zip(te.obj_color.flatten(), te.obj_shape.flatten())}
    assert seen_t <= held, f"evaluation scenes leaked non-held-out pairs: {seen_t - held}"


def test_too_small_a_split_raises_instead_of_leaking():
    """With fewer than 3 allowed pairs the old code left scene slots at their
    default (red, square), quietly putting a training object in the held out
    evaluation. It must fail loudly instead."""
    with pytest.raises(ValueError, match="allowed"):
        ReachEnv(4, seed=0, held_out_pairs={(0, 1), (1, 2)}, train=False)


def test_success_requires_being_close():
    e = ReachEnv(2, seed=0)
    e.agent = e.target_pos().clone()
    assert e.success().all()
    e.agent = e.target_pos() + SUCCESS_RADIUS * 2
    assert not e.success().any()


# --- demonstrations: the multimodality claim --------------------------------
def test_demonstrations_are_genuinely_bimodal():
    """The premise. Left and right modes must have opposite lateral sign, and
    their average must point into the obstacle, which is what defeats a
    regression head."""
    d = collect(400, seed=0)
    early = (torch.arange(d["action"].shape[0]) % 24) < 3
    ax = d["action"][early][:, 0, 0]
    side = d["side"][early]
    left, right = ax[side < 0].mean().item(), ax[side > 0].mean().item()
    assert left < -0.2 and right > 0.2, f"modes not separated: {left}, {right}"
    assert abs(ax.mean().item()) < OBSTACLE_HALF_W, \
        "the average of the two modes should point at the obstacle"


def test_both_sides_are_used():
    d = collect(400, seed=0)
    frac = (d["side"] > 0).float().mean().item()
    assert 0.35 < frac < 0.65, f"one mode dominates: {frac}"


def test_scripted_demonstrator_actually_solves_the_task():
    """The ceiling. If the demonstrator cannot do it, no head can."""
    assert collect(256, seed=3)["success_rate"] > 0.85


def test_waypoint_routes_go_opposite_ways():
    agent = torch.zeros(2, 2) - 0.5
    target = torch.tensor([[0.0, 0.72], [0.0, 0.72]])
    wp = waypoint_route(agent, target, torch.tensor([-1.0, 1.0]))
    assert wp[0, 0] < 0 < wp[1, 0]


# --- heads ------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(HEADS))
@pytest.mark.parametrize("chunk", [1, 4])
def test_head_shapes_and_finite_loss(name, chunk):
    torch.manual_seed(0)
    h = HEADS[name](128, chunk=chunk)
    feat = torch.randn(8, 128)
    act = torch.rand(8, chunk, 2) * 2 - 1
    loss = h.loss(feat, act)
    assert torch.isfinite(loss)
    assert h.sample(feat).shape == (8, chunk, 2)
    assert h.nfe() >= 1


@pytest.mark.parametrize("name", sorted(HEADS))
def test_sampled_actions_are_in_range(name):
    torch.manual_seed(0)
    s = HEADS[name](128)(torch.randn(16, 128)) if False else HEADS[name](128).sample(torch.randn(16, 128))
    assert s.abs().max() <= 1.0 + 1e-5


def test_regression_head_is_deterministic_and_the_others_are_not():
    """The property that decides everything else. A regression head returns the
    same action twice; the multimodal heads sample."""
    torch.manual_seed(0)
    feat = torch.randn(32, 128)
    r = RegressionHead(128)
    assert torch.allclose(r.sample(feat), r.sample(feat))
    for name in ("diffusion", "flow (pi-0 style)", "discrete bins"):
        h = HEADS[name](128)
        assert not torch.allclose(h.sample(feat), h.sample(feat)), f"{name} is deterministic"


def test_regression_learns_the_mean_of_two_modes():
    """The failure mode, isolated. Trained on +1 and -1 with equal probability,
    an MSE head returns approximately 0, which on this task is the obstacle."""
    torch.manual_seed(0)
    h = RegressionHead(4, act_dim=1)
    opt = torch.optim.Adam(h.parameters(), 1e-2)
    feat = torch.zeros(512, 4)
    for _ in range(400):
        target = torch.where(torch.rand(512, 1) < 0.5, -1.0, 1.0).unsqueeze(1)
        loss = h.loss(feat, target)
        opt.zero_grad(); loss.backward(); opt.step()
    assert h.sample(feat[:1]).abs().item() < 0.25


def test_flow_head_can_represent_two_modes():
    """The contrast: given the same data, a flow head puts mass on both."""
    torch.manual_seed(0)
    h = HEADS["flow (pi-0 style)"](4, act_dim=1)
    opt = torch.optim.Adam(h.parameters(), 1e-2)
    feat = torch.zeros(512, 4)
    for _ in range(1200):
        target = torch.where(torch.rand(512, 1) < 0.5, -1.0, 1.0).unsqueeze(1)
        loss = h.loss(feat, target)
        opt.zero_grad(); loss.backward(); opt.step()
    s = h.sample(torch.zeros(512, 4), steps=8).flatten()
    assert (s > 0.4).float().mean() > 0.2 and (s < -0.4).float().mean() > 0.2, \
        f"flow head collapsed to one mode: {s.mean():.3f}"


def test_discrete_bins_quantisation_floor():
    """21 bins over [-1, 1] is a bin width of 0.1, so the floor is half that."""
    h = DiscreteBins(128, bins=21)
    assert abs(h.quantisation_error - 0.05) < 1e-6
    assert h.quantise(torch.tensor([[0.0, 1.0]])).tolist() == [[10, 20]]


def test_nfe_reflects_sampling_cost():
    assert HEADS["regression"](128).nfe() == 1
    assert HEADS["flow (pi-0 style)"](128).nfe(5) == 5
    assert HEADS["diffusion"](128).nfe(50) == 50
    assert HEADS["discrete bins"](128, chunk=3).nfe() == 6


# --- encoder and eval -------------------------------------------------------
def test_encoder_uses_the_instruction():
    """Changing only the instruction must change the feature, otherwise the
    language input is decorative."""
    torch.manual_seed(0)
    enc = VisionLanguageEncoder()
    img = torch.rand(4, 3, 24, 24)
    st = torch.zeros(4, 2)
    a = enc(img, torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), st)
    b = enc(img, torch.ones(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), st)
    assert not torch.allclose(a, b)


def test_rollout_and_latency_report_sane_values():
    torch.manual_seed(0)
    enc = VisionLanguageEncoder()
    head = HEADS["regression"](enc.dim)
    r = rollout(enc, head, n=16, horizon=6, seed=0)
    assert 0.0 <= r["success"] <= 1.0 and r["blocked_steps"] >= 0
    lat = latency(enc, head, n=8, repeats=3)
    assert lat["latency_s"] > 0 and lat["max_hz"] > 0 and lat["nfe"] == 1


def test_trace_matches_the_rollout_it_came_from():
    """The animation draws these arrays next to numbers from the same call, so a
    trace that disagreed with its own blocked count would be a quiet lie."""
    torch.manual_seed(0)
    enc = VisionLanguageEncoder()
    head = HEADS["regression"](enc.dim)
    r = rollout(enc, head, n=16, horizon=6, seed=0, trace=True)
    assert r["path"].shape == (7, 16, 2)
    assert r["pressed"].shape == (6, 16)
    assert r["pressed"].float().sum(0).mean().item() == pytest.approx(r["blocked_steps"])
    assert r["path"].abs().max().item() <= 1.0
