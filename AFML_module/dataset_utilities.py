import numpy as np
import pandas as pd


def instrument_parser(instrument):
    """Parses contract names to expiration datetimes. Doesn't handle spreads"""

    month_map = {"H": 3, "M": 6, "U": 9, "Z": 12}  # March, June, April, September, December

    expiration_month = month_map[instrument[2]]
    expiration_year = int("202" + instrument[3])  # will not work beyond this decade; fix

    first_day_of_month = pd.Timestamp(year=expiration_year, month=expiration_month, day=1)
    first_friday_of_month = first_day_of_month + pd.DateOffset(days=(4 - first_day_of_month.weekday()) % 7)
    expiration_time = first_friday_of_month + pd.DateOffset(weeks=2) + pd.Timedelta(hours=22, minutes=0,
                                                                                    seconds=0)  # get expiration time
    return expiration_time


def get_instrument_attributes(instruments):
    """Given instruments and roll strategy (to do), builds summary frame"""

    instrument_frame = pd.DataFrame(columns=["symbol", "Expiration"],
                                    data=[[s, instrument_parser(s)] for s in instruments])

    instrument_frame = instrument_frame.sort_values(by="Expiration").reset_index(drop=True)

    # assumes that you roll a day and an hour before the instrument expires
    instrument_frame["Start"] = instrument_frame["Expiration"].shift() - pd.Timedelta(days=1, hours=1, minutes=0, seconds=0)
    instrument_frame["Stop"] = instrument_frame["Expiration"] - pd.Timedelta(days=1, hours=1, minutes=0, seconds=0)

    instrument_frame.at[0, "Start"] = pd.Timestamp(year=2023, month=1,
                                                   day=1)  # an arbitrary day before any strategy begins
    return instrument_frame


def form_dollar_bars(frame, thresh, sort):
    """Form bars of certain dollar size (actually index points, multiply by $50 to get dollars)"""

    frame = frame.copy()
    frame = frame.sort_values(by="ts_recv") if sort is True else frame

    frame["spend"] = frame["size"] * frame["price"]
    frame["symbol"] = frame["symbol"].astype("category")

    frame["cumulative spend"] = frame.groupby("symbol", observed=True)["spend"].cumsum()
    frame["dollar bin"] = (frame["cumulative spend"] // thresh).astype("int32")

    grouped = frame.groupby(["symbol", "dollar bin"], observed=True)

    agg_funcs = {
        "price": ["first", "last", "min", "max"],
        "size": "sum",
        "ts_recv": ["first", "last"],
        "spend": "sum",
    }

    df = grouped.agg(agg_funcs)
    df.columns = [
        "open", "close", "low", "high", "volume",
        "first transaction", "last transaction", "total spend"
    ]
    df["return"] = (df["close"] - df["open"]) / df["close"]
    df = df.reset_index()

    return df

def form_time_bars(frame, duration, sort):
    """Form time bars of duration"""

    frame = frame.copy()

    frame = frame.sort_values(by="ts_recv") if sort == True else frame
    frame["symbol"] = frame["symbol"].astype("category")

    # since we are using first, data must already be sorted
    frame["time difference"] = (frame["ts_recv"]
                                - frame.groupby("symbol", observed=True)["ts_recv"].transform("first"))

    frame["time bin"] = frame["time difference"].dt.floor(f"{duration}")

    grouped = frame.groupby(['symbol', pd.Grouper(key='ts_recv', freq=f'{duration}')], observed=True)

    agg_funcs = {
        "price": ["first", "last", "min", "max"],
        "size": ["sum", "count"],
        "ts_recv": ["first", "last"],
    }

    df = grouped.agg(agg_funcs)
    df.columns = [
        "open", "close", "low", "high",
        "volume", "ticks", "first transaction", "last transaction"
    ]

    df["return"] = (df["close"] - df["open"]) / df["close"]

    return df.reset_index()


def form_vol_bars(frame, thresh, sort):
    """Form volume bars of fixed volume size"""

    frame = frame.copy()
    frame = frame.sort_values(by="ts_recv") if sort is True else frame

    frame["symbol"] = frame["symbol"].astype("category")

    frame["cumulative size"] = frame.groupby("symbol", observed=True)["size"].cumsum()
    frame["vol bin"] = (frame["cumulative size"] // thresh).astype("int32")

    grouped = frame.groupby(["symbol", "vol bin"], observed=True)

    agg_funcs = {
        "price": ["first", "last", "min", "max"],
        "size": "sum",
        "ts_recv": ["first", "last"],
    }

    df = grouped.agg(agg_funcs)
    df.columns = [
        "open", "close", "low", "high",
        "volume", "first transaction", "last transaction"
    ]
    df["return"] = (df["close"] - df["open"]) / df["close"]

    return df.reset_index()


def reduce_to_active_symbols(bars, instrument_frame):
    frame = bars.copy()

    # determine active symbol at each time
    for idx, row in instrument_frame.iterrows():
        symbol = row["symbol"]
        start = row["Start"]
        stop = row["Stop"]

        frame.loc[((frame["first transaction"] >= start) & (frame["last transaction"] < stop)), "Active Symbol"] = symbol

    frame = frame[frame["symbol"] == frame["Active Symbol"]].reset_index(drop=True)

    return frame


def apply_roll_factors(frame, sort):

    frame = frame.copy().sort_values(by="time start").reset_index() if sort is True else frame

    # get factors to transition from one symbol to next
    frame["Factor"] = 1.0
    new_symbol_indices = frame[frame["Active Symbol"] != frame["Active Symbol"].shift(1)].index

    frame.at[new_symbol_indices[0], "Factor"] = 1  # /frame.at[new_symbol_indices[0], "open"]

    # this for loop is not that big of a deal if we aren't rolling too often
    for new_sym in new_symbol_indices[1:]:
        frame.at[new_sym, "Factor"] = frame.at[new_sym - 1, "close"] / frame.at[new_sym, "open"]

    frame["Factor"] = frame["Factor"].cumprod()

    frame["open"] = frame["open"] * frame["Factor"]
    frame["close"] = frame["close"] * frame["Factor"]
    frame["low"] = frame["low"] * frame["Factor"]
    frame["high"] = frame["high"] * frame["Factor"]

    return frame

