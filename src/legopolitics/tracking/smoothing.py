from __future__ import annotations


def exponential_smooth(values: list[float], alpha: float = 0.4) -> list[float]:
    if not values:
        return []
    output = [values[0]]
    for value in values[1:]:
        output.append(alpha * value + (1 - alpha) * output[-1])
    return output
