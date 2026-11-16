"""The installed SmolVLA noise generator must replay after an explicit episode reset."""
from types import SimpleNamespace
import pytest
torch = pytest.importorskip('torch')
from lerobot.policies.common.flow_matching import sample_noise
from explorer3.policy_runtime import PolicyCore


def test_explicit_reset_replays_flow_noise_without_previous_episode_leakage():
    core = PolicyCore.__new__(PolicyCore)
    core.torch, core.device = torch, 'cpu'
    core.policy = SimpleNamespace(reset=lambda: None)
    first = core.reset(11000)
    expected = sample_noise((1, 50, 32), 'cpu')
    sample_noise((13, 50, 32), 'cpu')  # another episode consumes a different amount
    second = core.reset(11000)
    assert first == second
    assert torch.equal(expected, sample_noise((1, 50, 32), 'cpu'))
    core.reset(11001)
    assert not torch.equal(expected, sample_noise((1, 50, 32), 'cpu'))
    assert core.reset()['seed'] is None


@pytest.mark.parametrize('seed', [-1, 2**32, True, 1.5])
def test_invalid_sampling_seed_is_rejected(seed):
    core = PolicyCore.__new__(PolicyCore)
    with pytest.raises(ValueError, match='sampling seed'):
        core.reset(seed)
