import numpy as np
import pandas as pd

def apply_cusum_filter(bar_frame, h, col="return"):
    """Given bar frame and threshold, returns indices
    of sample points."""

    bar_frame[col] = bar_frame[col].fillna(0) # no change to s if value is none

    diff = bar_frame[col].diff().to_numpy()

    s_plus = np.zeros(diff.shape[0])
    s_minus = np.zeros(diff.shape[0])

    events = []

    for i in range(1, len(diff)):
        s_plus[i] = max(0, s_plus[i - 1] + diff[i])
        s_minus[i] = min(0, s_minus[i - 1] + diff[i])

        if s_plus[i] >= h:
            events.append(i)
            s_plus[i] = 0

        elif s_minus[i] <= -h:
            events.append(i)
            s_minus[i] = 0

    return events