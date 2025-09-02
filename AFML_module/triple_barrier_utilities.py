import numpy as np
import pandas as pd

def get_daily_rolling_volatility(close_frame):
    """ For each entry in close_frame, computes return over prior day.
    Compute rolling standard deviation of these returns """

    day_prior_positions = close_frame["last transaction"].searchsorted(close_frame["last transaction"] - pd.Timedelta("1d"))

    day_prior_positions = day_prior_positions[day_prior_positions > 0]

    day_prior_mapper = pd.Series(close_frame.index[day_prior_positions - 1],
                                 index=close_frame.index[close_frame.shape[0] - day_prior_positions.shape[0]:])

    rolling_vol = close_frame.loc[day_prior_mapper.index]["close"] / close_frame.loc[day_prior_mapper.values]["close"].values - 1

    rolling_vol = rolling_vol.rename("return").to_frame()
    rolling_vol["daily volatility"] = rolling_vol["return"].ewm(span=10).std()

    rolling_vol = rolling_vol.dropna()

    return rolling_vol

def get_vertical_barriers(close_frame, time_thresh):
    """ For every entry in close_frame, finds index of nearest entry
    occuring >= time_thresh later. """

    next_period = close_frame["last transaction"].searchsorted(close_frame["last transaction"] + pd.Timedelta(time_thresh))
    next_period = next_period[next_period < close_frame.shape[0]]

    time_barriers = pd.Series(close_frame.index[next_period], index=close_frame.index[:next_period.shape[0]])

    return time_barriers


def detect_crossed_barriers_for_events(close_frame,
                                       events,
                                       lower_series,
                                       upper_series,
                                       lower_factor,
                                       upper_factor,
                                       vertical_barriers
                                       ):
    """ Each event markets left edge of a box. For each box, find the barrier
    of first intersection. Return barrier, absolute return, and index of
    first touch. """

    output = pd.DataFrame(index=events, columns=["barrier", "absolute return", "first touch"])

    lower_barriers = close_frame["close"] * (1 - lower_series * lower_factor)
    upper_barriers = close_frame["close"] * (1 + upper_series * upper_factor)

    # For each box, find which wall is crossed first
    for event in events:

        # barriers may not be defined at start/end of series
        try:
            vertical_barrier = vertical_barriers[event]
            box_lower_barrier = lower_barriers[event]
            box_upper_barrier = upper_barriers[event]
        except KeyError:
            continue

        box_prices = close_frame.loc[event: vertical_barrier]

        lower_touches = box_prices[box_prices["close"] <= box_lower_barrier]
        upper_touches = box_prices[box_prices["close"] >= box_upper_barrier]

        lower_touch_min = lower_touches.index[0] if len(lower_touches) > 0 else np.inf
        upper_touch_min = upper_touches.index[0] if len(upper_touches) > 0 else np.inf

        # Return the side, size, and index of the first touch.
        first_close = box_prices.iloc[0]["close"]

        if lower_touch_min == np.inf and upper_touch_min == np.inf:
            output.loc[event, "barrier"] = 0
            output.loc[event, "absolute return"] = np.abs(box_prices.iloc[-1]["close"] / first_close - 1)
            output.loc[event, "first touch"] = box_prices.index[-1]

        elif lower_touch_min < upper_touch_min:
            output.loc[event, "barrier"] = -1
            output.loc[event, "absolute return"] = np.abs(lower_touches.iloc[0]["close"] / first_close - 1)
            output.loc[event, "first touch"] = lower_touches.index[0]

        elif upper_touch_min < lower_touch_min:
            output.loc[event, "barrier"] = 1
            output.loc[event, "absolute return"] = np.abs(upper_touches.iloc[0]["close"] / first_close - 1)
            output.loc[event, "first touch"] = upper_touches.index[0]

    return output
