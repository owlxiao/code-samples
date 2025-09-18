import torch


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    # read MN elements; write MN elements
    numerator = x.exp()

    # read MN elements; write M elements
    denominator = numerator.sum(dim=-1, keepdim=True)

    # read MN + M elements; write MN elements
    ret = numerator / denominator[:, None]

    # in total: read 3MN + M elements; write 2MN + M elements
    return ret


def safe_softmax(x: torch.Tensor) -> torch.Tensor:
    # read  MN elements ; write M  elements
    x_max = x.max(dim=1)[0]

    # read MN + M elements ; write MN elements
    z = x - x_max[:, None]

    # read  MN elements ; write MN elements
    numerator = torch.exp(z)

    # read  MN elements ; write M  elements
    denominator = numerator.sum(dim=1)

    # read MN + M elements ; write MN elements
    ret = numerator / denominator[:, None]

    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    return ret
