import math

def compute_entropy(probs):
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy

