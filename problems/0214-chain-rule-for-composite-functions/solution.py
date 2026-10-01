import torch


def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    value = torch.tensor(x, dtype=torch.float64)       # current output of the pipeline
    derivative = torch.tensor(1.0, dtype=torch.float64)  # running product of local derivatives

    for name in reversed(functions):   # applied right to left
        if name == 'square':
            derivative = derivative * 2 * value
            value = value ** 2
        elif name == 'sin':
            derivative = derivative * torch.cos(value)
            value = torch.sin(value)
        elif name == 'exp':
            derivative = derivative * torch.exp(value)
            value = torch.exp(value)
        elif name == 'log':
            derivative = derivative / value
            value = torch.log(value)
        else:
            raise ValueError(f"Unknown function: {name}")

    return derivative.item()