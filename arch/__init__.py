from .basics import *  # noqa
from .mnas import *  # noqa


class _SplitEval(torch.nn.Module):
    def __init__(self, f, size):
        super().__init__()
        self.f = f
        self.size = size

    def forward(self, x):
        return torch.cat([self.f(x[i: i + self.size]) for i in range(0, len(x), self.size)])


def init_arch(datasets, **args):
    torch.manual_seed(0)

    assert datasets[0].shape[0] == args['ptr']

    if args['act'] == 'relu':
        _act = torch.relu
    elif args['act'] == 'tanh':
        _act = torch.tanh
    elif args['act'] == 'softplus':
        _act = torch.nn.functional.softplus
    elif args['act'] == 'swish':
        _act = torch.nn.functional.silu
    else:
        raise ValueError('act not specified')

    def __act(x):
        b = args['act_beta']
        return _act(b * x) / b
    factor = __act(torch.randn(100000, dtype=torch.float64)).pow(2).mean().rsqrt().item()

    def act(x):
        b = args['act_beta']
        return _act(b * x).mul_(factor / b)

    _d = abs(act(torch.randn(100000, dtype=torch.float64)).pow(2).mean().rsqrt().item() - 1)
    assert _d < 1e-2, _d

    torch.manual_seed(args['seed_init'])

    if args['arch'] == 'fc':
        assert args['L'] is not None
        datasets = [x.flatten(1) for x in datasets]
        f = FC(datasets[0].size(1), args['h'], 1, args['L'], act, args['bias'], args['last_bias'], args['var_bias'])

    elif args['arch'] == 'cv':
        assert args['bias'] == 0
        f = CV(datasets[0].size(1), args['h'], L1=args['cv_L1'], L2=args['cv_L2'], act=act, h_base=args['cv_h_base'],
               fsz=args['cv_fsz'], pad=args['cv_pad'], stride_first=args['cv_stride_first'])
    elif args['arch'] == 'resnet':
        assert args['bias'] == 0
        f = Wide_ResNet(datasets[0].size(1), 28, args['h'], act, 1, args['mix_angle'])
    elif args['arch'] == 'mnas':
        f = MnasNetLike(datasets[0].size(1), args['h'], 1, args['cv_L1'], args['cv_L2'], act=act, dim=datasets[0].dim() - 2)
    elif args['arch'] == 'mnist':
        assert args['dataset'] == 'mnist'
        f = MNISTNet(datasets[0].size(1), args['h'], 1, act)
    elif args['arch'] == 'fixed_weights':
        f = FixedWeights(args['d'], args['h'], act, args['bias'])
    elif args['arch'] == 'fixed_angles':
        f = FixedAngles(args['d'], args['h'], act, args['bias'])
    elif args['arch'] == 'conv1d':
        f = Conv1d(args['d'], args['h'], act, args['bias'])
    else:
        raise ValueError('arch not specified')

    f = f.to(args['device'])
    f = _SplitEval(f, args['chunk'])

    return f, datasets
