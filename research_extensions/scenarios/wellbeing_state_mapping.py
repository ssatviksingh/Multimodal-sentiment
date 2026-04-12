"""
wellbeing_state_mapping.py

Utilities to map raw sentiment predictions from the multimodal model
into higher-level wellbeing / risk states suitable for telehealth.

We interpret sentiment over time and derive:
- Calm
- Mild Concern
- High Concern
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple
import numpy as np

SentimentLabel = Literal["negative", "neutral", "positive"]
RiskState = Literal["Calm", "Mild Concern", "High Concern"]


@dataclass
class WindowPrediction:
    """Container for a single time window prediction."""

    start_time: float  # seconds
    end_time: float    # seconds
    sentiment: SentimentLabel
    confidence: float  # softmax confidence or probability in [0, 1]


def map_sentiment_to_base_state(sentiment: SentimentLabel) -> RiskState:
    """
    Direct mapping from instantaneous sentiment to a coarse wellbeing state.
    This is later refined using temporal trends.
    """
    if sentiment == "positive":
        return "Calm"
    if sentiment == "neutral":
        # Neutral can be genuinely calm or possible masked distress;
        # temporal trends and cross-modal agreement will refine this.
        return "Mild Concern"
    # sentiment == "negative"
    return "High Concern"


def smooth_risk_states(
    states: List[RiskState],
    window: int = 3,
) -> List[RiskState]:
    """
    Simple temporal smoothing over discrete risk states using
    a majority vote in a local window.

    Args:
        states: list of raw risk states per window.
        window: odd window size for smoothing (e.g., 3 or 5).
    """
    if window <= 1 or len(states) <= 1:
        return states

    k = window // 2
    smoothed: List[RiskState] = []
    for i in range(len(states)):
        left = max(0, i - k)
        right = min(len(states), i + k + 1)
        segment = states[left:right]
        # majority vote with priority High > Mild > Calm in case of ties
        counts = {s: segment.count(s) for s in set(segment)}
        # enforce priority ordering
        for level in ["High Concern", "Mild Concern", "Calm"]:
            if counts.get(level, 0) == max(counts.values()):
                smoothed.append(level)  # type: ignore[arg-type]
                break
    return smoothed


def derive_risk_sequence(
    preds: List[WindowPrediction],
    high_thresh: float = 0.6,
    mild_thresh: float = 0.4,
    smooth_window: int = 3,
) -> List[RiskState]:
    """
    Convert a sequence of window-level sentiment predictions into
    a smoothed sequence of wellbeing / risk states.

    Heuristic:
    - If sentiment is negative and confidence >= high_thresh → High Concern
    - If sentiment is negative but confidence < high_thresh → Mild Concern
    - If sentiment is neutral:
        - confidence >= high_thresh → Mild Concern
        - else → Calm (assume ambiguous)
    - If sentiment is positive:
        - confidence >= mild_thresh → Calm
        - else → Mild Concern (uncertain positivity)
    """
    raw_states: List[RiskState] = []
    for wp in preds:
        s, c = wp.sentiment, float(wp.confidence)
        if s == "negative":
            if c >= high_thresh:
                raw_states.append("High Concern")
            else:
                raw_states.append("Mild Concern")
        elif s == "neutral":
            if c >= high_thresh:
                raw_states.append("Mild Concern")
            else:
                raw_states.append("Calm")
        else:  # positive
            if c >= mild_thresh:
                raw_states.append("Calm")
            else:
                raw_states.append("Mild Concern")

    return smooth_risk_states(raw_states, window=smooth_window)


def summarize_risk(states: List[RiskState]) -> Tuple[float, float, float]:
    """
    Compute percentage of time spent in each risk state over a session.
    Assumes each window has equal duration.
    """
    if not states:
        return 0.0, 0.0, 0.0

    n = len(states)
    calm = states.count("Calm") / n * 100.0
    mild = states.count("Mild Concern") / n * 100.0
    high = states.count("High Concern") / n * 100.0
    return calm, mild, high

