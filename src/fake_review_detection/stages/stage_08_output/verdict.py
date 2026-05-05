# Maps model P(genuine) to short labels the UI can show without staring at raw probabilities.
def verdict_from_probability(p_genuine: float) -> tuple[str, str]:
    p = float(p_genuine)
    if p >= 0.82:
        return "Genuine (strong)", "P(genuine) ≥ 0.82 — high confidence the review matches genuine patterns in the training data."
    if p >= 0.62:
        return "Genuine (moderate)", "P(genuine) between 0.62 and 0.82 — leans genuine but not decisive."
    if p >= 0.5:
        return "Genuine (weak)", "P(genuine) just above 0.5 — slight lean toward genuine."
    if p >= 0.38:
        return "Deceptive (weak)", "P(genuine) just below 0.5 — slight lean toward deceptive."
    if p >= 0.18:
        return "Deceptive (moderate)", "P(genuine) between 0.18 and 0.38 — leans deceptive."
    return "Deceptive (strong)", "P(genuine) ≤ 0.18 — high confidence the review matches deceptive patterns in the training data."


def binary_class_from_probability(p_genuine: float) -> bool:
    return p_genuine >= 0.5
