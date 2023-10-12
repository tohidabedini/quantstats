#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2023 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from warnings import warn
import pandas as _pd
import numpy as _np
from math import ceil as _ceil, sqrt as _sqrt
from scipy.stats import norm as _norm, linregress as _linregress

from . import utils as _utils

from .reports import PERIODS_PER_YEAR


# ======== STATS ========


def pct_rank(prices, window=60):
    """Rank prices by window"""
    rank = _utils.multi_shift(prices, window).T.rank(pct=True).T
    return rank.iloc[:, 0] * 100.0


def compsum(returns):
    """Calculates rolling compounded returns"""
    return returns.add(1).cumprod() - 1


def comp(returns):
    """Calculates total compounded returns"""
    return returns.add(1).prod() - 1


def distribution(returns, compounded=True, prepare_returns=True):
    def get_outliers(data):
        # https://datascience.stackexchange.com/a/57199
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1  # IQR is interquartile range.
        filtered = (data >= Q1 - 1.5 * IQR) & (data <= Q3 + 1.5 * IQR)
        return {
            "values": data.loc[filtered].tolist(),
            "outliers": data.loc[~filtered].tolist(),
        }

    if isinstance(returns, _pd.DataFrame):
        warn(
            "Pandas DataFrame was passed (Series expected). "
            "Only first column will be used."
        )
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and "close" in returns.columns:
            returns = returns["close"]
        else:
            returns = returns[returns.columns[0]]

    apply_fnc = comp if compounded else _np.sum
    daily = returns.dropna()

    if prepare_returns:
        daily = _utils._prepare_returns(daily)

    return {
        "Daily": get_outliers(daily),
        "Weekly": get_outliers(daily.resample("W-MON").apply(apply_fnc)),
        "Monthly": get_outliers(daily.resample("M").apply(apply_fnc)),
        "Quarterly": get_outliers(daily.resample("Q").apply(apply_fnc)),
        "Yearly": get_outliers(daily.resample("A").apply(apply_fnc)),
    }


def expected_return(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Returns the expected return for a given period
    by calculating the geometric holding period return
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return _np.product(1 + returns) ** (1 / len(returns)) - 1


def geometric_mean(retruns, aggregate=None, compounded=True):
    """Shorthand for expected_return()"""
    return expected_return(retruns, aggregate, compounded)


def ghpr(retruns, aggregate=None, compounded=True):
    """Shorthand for expected_return()"""
    return expected_return(retruns, aggregate, compounded)


def outliers(returns, quantile=0.95):
    """Returns series of outliers"""
    return returns[returns > returns.quantile(quantile)].dropna(how="all")


def remove_outliers(returns, quantile=0.95):
    """Returns series of returns without the outliers"""
    return returns[returns < returns.quantile(quantile)]


def best(returns, aggregate=None, compounded=True, prepare_returns=True):
    """Returns the best day/month/week/quarter/year's return"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return _utils.aggregate_returns(returns, aggregate, compounded).max()


def worst(returns, aggregate=None, compounded=True, prepare_returns=True):
    """Returns the worst day/month/week/quarter/year's return"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return _utils.aggregate_returns(returns, aggregate, compounded).min()


def consecutive_wins(returns, aggregate=None, compounded=True, prepare_returns=True):
    """Returns the maximum consecutive wins by day/month/week/quarter/year"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    returns = _utils.aggregate_returns(returns, aggregate, compounded) > 0
    return _utils._count_consecutive(returns).max()


def consecutive_losses(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Returns the maximum consecutive losses by
    day/month/week/quarter/year
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    returns = _utils.aggregate_returns(returns, aggregate, compounded) < 0
    return _utils._count_consecutive(returns).max()


def exposure(returns, prepare_returns=True):
    """Returns the market exposure time (returns != 0)"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    def _exposure(ret):
        ex = len(ret[(~_np.isnan(ret)) & (ret != 0)]) / len(ret)
        return _ceil(ex * 100) / 100

    if isinstance(returns, _pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _exposure(returns[col])
        return _pd.Series(_df)
    return _exposure(returns)


def win_rate(returns, aggregate=None, compounded=True, prepare_returns=True):
    """Calculates the win ratio for a period"""

    def _win_rate(series):
        try:
            return len(series[series > 0]) / len(series[series != 0])
        except Exception:
            return 0.0

    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)

    if isinstance(returns, _pd.DataFrame):
        _df = {}
        for col in returns.columns:
            _df[col] = _win_rate(returns[col])

        return _pd.Series(_df)

    return _win_rate(returns)


def avg_return(returns, aggregate=None, compounded=True, prepare_returns=True):
    """Calculates the average return/trade return for a period"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns != 0].dropna().mean()


def avg_win(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Calculates the average winning
    return/trade return for a period
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns > 0].dropna().mean()


def avg_loss(returns, aggregate=None, compounded=True, prepare_returns=True):
    """
    Calculates the average low if
    return/trade return for a period
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    if aggregate:
        returns = _utils.aggregate_returns(returns, aggregate, compounded)
    return returns[returns < 0].dropna().mean()


def volatility(returns, periods=PERIODS_PER_YEAR, annualize=True, prepare_returns=True):
    """Calculates the volatility of returns for a period"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    std = returns.std()
    if annualize:
        return std * _np.sqrt(periods)

    return std


def rolling_volatility(
    returns, rolling_period=126, periods_per_year=PERIODS_PER_YEAR, prepare_returns=True
):
    if prepare_returns:
        returns = _utils._prepare_returns(returns, rolling_period)

    return returns.rolling(rolling_period).std() * _np.sqrt(periods_per_year)


def implied_volatility(returns, periods=PERIODS_PER_YEAR, annualize=True):
    """Calculates the implied volatility of returns for a period"""
    logret = _utils.log_returns(returns)
    if annualize:
        return logret.rolling(periods).std() * _np.sqrt(periods)
    return logret.std()


def autocorr_penalty(returns, prepare_returns=False):
    """Metric to account for auto correlation"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)

    if isinstance(returns, _pd.DataFrame):
        returns = returns[returns.columns[0]]

    # returns.to_csv('/Users/ran/Desktop/test.csv')
    num = len(returns)
    coef = _np.abs(_np.corrcoef(returns[:-1], returns[1:])[0, 1])
    corr = [((num - x) / num) * coef**x for x in range(1, num)]
    return _np.sqrt(1 + 2 * _np.sum(corr))


# ======= METRICS =======


def sharpe(returns, rf=0.0, periods=PERIODS_PER_YEAR, annualize=True, smart=False):
    """
    Calculates the sharpe ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Args:
        * returns (Series, DataFrame): Input return series
        * rf (float): Risk-free rate expressed as a yearly (annualized) return
        * periods (int): Freq. of returns (PERIODS_PER_YEAR/365 for daily, 12 for monthly)
        * annualize: return annualize sharpe?
        * smart: return smart sharpe ratio
    """
    if rf != 0 and periods is None:
        raise Exception("Must provide periods if rf != 0")

    returns = _utils._prepare_returns(returns, rf, periods)
    divisor = returns.std(ddof=1)
    if smart:
        # penalize sharpe with auto correlation
        divisor = divisor * autocorr_penalty(returns)
    res = returns.mean() / divisor

    if annualize:
        return res * _np.sqrt(1 if periods is None else periods)

    return res


def smart_sharpe(returns, rf=0.0, periods=PERIODS_PER_YEAR, annualize=True):
    return sharpe(returns, rf, periods, annualize, True)


def rolling_sharpe(
    returns,
    rf=0.0,
    rolling_period=126,
    annualize=True,
    periods_per_year=PERIODS_PER_YEAR,
    prepare_returns=True,
):

    if rf != 0 and rolling_period is None:
        raise Exception("Must provide periods if rf != 0")

    if prepare_returns:
        returns = _utils._prepare_returns(returns, rf, rolling_period)

    res = returns.rolling(rolling_period).mean() / returns.rolling(rolling_period).std()

    if annualize:
        res = res * _np.sqrt(1 if periods_per_year is None else periods_per_year)
    return res


def sortino(returns, rf=0, periods=PERIODS_PER_YEAR, annualize=True, smart=False):
    """
    Calculates the sortino ratio of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Calculation is based on this paper by Red Rock Capital
    http://www.redrockcapital.com/Sortino__A__Sharper__Ratio_Red_Rock_Capital.pdf
    """
    if rf != 0 and periods is None:
        raise Exception("Must provide periods if rf != 0")

    returns = _utils._prepare_returns(returns, rf, periods)

    downside = _np.sqrt((returns[returns < 0] ** 2).sum() / len(returns))

    if smart:
        # penalize sortino with auto correlation
        downside = downside * autocorr_penalty(returns)

    res = returns.mean() / downside

    if annualize:
        return res * _np.sqrt(1 if periods is None else periods)

    return res


def smart_sortino(returns, rf=0, periods=PERIODS_PER_YEAR, annualize=True):
    return sortino(returns, rf, periods, annualize, True)


def rolling_sortino(
    returns, rf=0, rolling_period=126, annualize=True, periods_per_year=PERIODS_PER_YEAR, **kwargs
):
    if rf != 0 and rolling_period is None:
        raise Exception("Must provide periods if rf != 0")

    if kwargs.get("prepare_returns", True):
        returns = _utils._prepare_returns(returns, rf, rolling_period)

    downside = (
        returns.rolling(rolling_period).apply(
            lambda x: (x.values[x.values < 0] ** 2).sum()
        )
        / rolling_period
    )

    res = returns.rolling(rolling_period).mean() / _np.sqrt(downside)
    if annualize:
        res = res * _np.sqrt(1 if periods_per_year is None else periods_per_year)
    return res


def adjusted_sortino(returns, rf=0, periods=PERIODS_PER_YEAR, annualize=True, smart=False):
    """
    Jack Schwager's version of the Sortino ratio allows for
    direct comparisons to the Sharpe. See here for more info:
    https://archive.is/wip/2rwFW
    """
    data = sortino(returns, rf, periods=periods, annualize=annualize, smart=smart)
    return data / _sqrt(2)


def probabilistic_ratio(
    series, rf=0.0, base="sharpe", periods=PERIODS_PER_YEAR, annualize=False, smart=False
):

    if base.lower() == "sharpe":
        base = sharpe(series, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "sortino":
        base = sortino(series, periods=periods, annualize=False, smart=smart)
    elif base.lower() == "adjusted_sortino":
        base = adjusted_sortino(series, periods=periods, annualize=False, smart=smart)
    else:
        raise Exception(
            "`metric` must be either `sharpe`, `sortino`, or `adjusted_sortino`"
        )
    skew_no = skew(series, prepare_returns=False)
    kurtosis_no = kurtosis(series, prepare_returns=False)

    n = len(series)

    sigma_sr = _np.sqrt(
        (
            1
            + (0.5 * base**2)
            - (skew_no * base)
            + (((kurtosis_no - 3) / 4) * base**2)
        )
        / (n - 1)
    )

    ratio = (base - rf) / sigma_sr
    psr = _norm.cdf(ratio)

    if annualize:
        return psr * (PERIODS_PER_YEAR**0.5)
    return psr


def probabilistic_sharpe_ratio(
    series, rf=0.0, periods=PERIODS_PER_YEAR, annualize=False, smart=False
):
    return probabilistic_ratio(
        series, rf, base="sharpe", periods=periods, annualize=annualize, smart=smart
    )


def probabilistic_sortino_ratio(
    series, rf=0.0, periods=PERIODS_PER_YEAR, annualize=False, smart=False
):
    return probabilistic_ratio(
        series, rf, base="sortino", periods=periods, annualize=annualize, smart=smart
    )


def probabilistic_adjusted_sortino_ratio(
    series, rf=0.0, periods=PERIODS_PER_YEAR, annualize=False, smart=False
):
    return probabilistic_ratio(
        series,
        rf,
        base="adjusted_sortino",
        periods=periods,
        annualize=annualize,
        smart=smart,
    )


def treynor_ratio(returns, benchmark, periods=float(PERIODS_PER_YEAR), rf=0.0):
    """
    Calculates the Treynor ratio

    Args:
        * returns (Series, DataFrame): Input return series
        * benchmatk (String, Series, DataFrame): Benchmark to compare beta to
        * periods (int): Freq. of returns (PERIODS_PER_YEAR/365 for daily, 12 for monthly)
    """
    if isinstance(returns, _pd.DataFrame):
        returns = returns[returns.columns[0]]

    beta = greeks(returns, benchmark, periods=periods).to_dict().get("beta", 0)
    if beta == 0:
        return 0
    return (comp(returns) - rf) / beta


def omega(returns, rf=0.0, required_return=0.0, periods=PERIODS_PER_YEAR):
    """
    Determines the Omega ratio of a strategy.
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.
    """
    if len(returns) < 2:
        return _np.nan

    if required_return <= -1:
        return _np.nan

    returns = _utils._prepare_returns(returns, rf, periods)

    if periods == 1:
        return_threshold = required_return
    else:
        return_threshold = (1 + required_return) ** (1.0 / periods) - 1

    returns_less_thresh = returns - return_threshold
    numer = returns_less_thresh[returns_less_thresh > 0.0].sum().values[0]
    denom = -1.0 * returns_less_thresh[returns_less_thresh < 0.0].sum().values[0]

    if denom > 0.0:
        return numer / denom

    return _np.nan


def gain_to_pain_ratio(returns, rf=0, resolution="D"):
    """
    Jack Schwager's GPR. See here for more info:
    https://archive.is/wip/2rwFW
    """
    returns = _utils._prepare_returns(returns, rf).resample(resolution).sum()
    downside = abs(returns[returns < 0].sum())
    return returns.sum() / downside


def cagr(returns, rf=0.0, compounded=True, periods=PERIODS_PER_YEAR):
    """
    Calculates the communicative annualized growth return
    (CAGR%) of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """
    total = _utils._prepare_returns(returns, rf)
    if compounded:
        total = comp(total)
    else:
        total = _np.sum(total)

    years = (returns.index[-1] - returns.index[0]).days / periods

    res = abs(total + 1.0) ** (1.0 / years) - 1

    if isinstance(returns, _pd.DataFrame):
        res = _pd.Series(res)
        res.index = returns.columns

    return res


def rar(returns, rf=0.0):
    """
    Calculates the risk-adjusted return of access returns
    (CAGR / exposure. takes time into account.)

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms
    """
    returns = _utils._prepare_returns(returns, rf)
    return cagr(returns) / exposure(returns)


def skew(returns, prepare_returns=True):
    """
    Calculates returns' skewness
    (the degree of asymmetry of a distribution around its mean)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return returns.skew()


def kurtosis(returns, prepare_returns=True):
    """
    Calculates returns' kurtosis
    (the degree to which a distribution peak compared to a normal distribution)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return returns.kurtosis()


def calmar(returns, prepare_returns=True):
    """Calculates the calmar ratio (CAGR% / MaxDD%)"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    cagr_ratio = cagr(returns)
    max_dd = max_drawdown(returns)
    return cagr_ratio / abs(max_dd)


def ulcer_index(returns):
    """Calculates the ulcer index score (downside risk measurment)"""
    dd = to_drawdown_series(returns)
    return _np.sqrt(_np.divide((dd**2).sum(), returns.shape[0] - 1))


def ulcer_performance_index(returns, rf=0):
    """
    Calculates the ulcer index score
    (downside risk measurment)
    """
    return (comp(returns) - rf) / ulcer_index(returns)


def upi(returns, rf=0):
    """Shorthand for ulcer_performance_index()"""
    return ulcer_performance_index(returns, rf)


def serenity_index(returns, rf=0):
    """
    Calculates the serenity index score
    (https://www.keyquant.com/Download/GetFile?Filename=%5CPublications%5CKeyQuant_WhitePaper_APT_Part1.pdf)
    """
    dd = to_drawdown_series(returns)
    pitfall = -cvar(dd) / returns.std()
    return (returns.sum() - rf) / (ulcer_index(returns) * pitfall)


def risk_of_ruin(returns, prepare_returns=True):
    """
    Calculates the risk of ruin
    (the likelihood of losing all one's investment capital)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    wins = win_rate(returns)
    return ((1 - wins) / (1 + wins)) ** len(returns)


def ror(returns):
    """Shorthand for risk_of_ruin()"""
    return risk_of_ruin(returns)


def value_at_risk(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """
    Calculats the daily value-at-risk
    (variance-covariance calculation with confidence n)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    mu = returns.mean()
    sigma *= returns.std()

    if confidence > 1:
        confidence = confidence / 100

    return _norm.ppf(1 - confidence, mu, sigma)


def var(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """Shorthand for value_at_risk()"""
    return value_at_risk(returns, sigma, confidence, prepare_returns)


def conditional_value_at_risk(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """
    Calculats the conditional daily value-at-risk (aka expected shortfall)
    quantifies the amount of tail risk an investment
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    var = value_at_risk(returns, sigma, confidence)
    c_var = returns[returns < var].values.mean()
    return c_var if ~_np.isnan(c_var) else var


def cvar(returns, sigma=1, confidence=0.95, prepare_returns=True):
    """Shorthand for conditional_value_at_risk()"""
    return conditional_value_at_risk(returns, sigma, confidence, prepare_returns)


def expected_shortfall(returns, sigma=1, confidence=0.95):
    """Shorthand for conditional_value_at_risk()"""
    return conditional_value_at_risk(returns, sigma, confidence)


def tail_ratio(returns, cutoff=0.95, prepare_returns=True):
    """
    Measures the ratio between the right
    (95%) and left tail (5%).
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return abs(returns.quantile(cutoff) / returns.quantile(1 - cutoff))


def payoff_ratio(returns, prepare_returns=True):
    """Measures the payoff ratio (average win/average loss)"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return avg_win(returns) / abs(avg_loss(returns))


def win_loss_ratio(returns, prepare_returns=True):
    """Shorthand for payoff_ratio()"""
    return payoff_ratio(returns, prepare_returns)


def profit_ratio(returns, prepare_returns=True):
    """Measures the profit ratio (win ratio / loss ratio)"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    wins = returns[returns >= 0]
    loss = returns[returns < 0]

    win_ratio = abs(wins.mean() / wins.count())
    loss_ratio = abs(loss.mean() / loss.count())
    try:
        return win_ratio / loss_ratio
    except Exception:
        return 0.0


def profit_factor(returns, prepare_returns=True):
    """Measures the profit ratio (wins/loss)"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return abs(returns[returns >= 0].sum() / returns[returns < 0].sum())


def cpc_index(returns, prepare_returns=True):
    """
    Measures the cpc ratio
    (profit factor * win % * win loss ratio)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return profit_factor(returns) * win_rate(returns) * win_loss_ratio(returns)


def common_sense_ratio(returns, prepare_returns=True):
    """Measures the common sense ratio (profit factor * tail ratio)"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return profit_factor(returns) * tail_ratio(returns)


def outlier_win_ratio(returns, quantile=0.99, prepare_returns=True):
    """
    Calculates the outlier winners ratio
    99th percentile of returns / mean positive return
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return returns.quantile(quantile).mean() / returns[returns >= 0].mean()


def outlier_loss_ratio(returns, quantile=0.01, prepare_returns=True):
    """
    Calculates the outlier losers ratio
    1st percentile of returns / mean negative return
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return returns.quantile(quantile).mean() / returns[returns < 0].mean()


def recovery_factor(returns, rf=0., prepare_returns=True):
    """Measures how fast the strategy recovers from drawdowns"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    total_returns = returns.sum() - rf
    max_dd = max_drawdown(returns)
    return abs(total_returns) / abs(max_dd)


def risk_return_ratio(returns, prepare_returns=True):
    """
    Calculates the return / risk ratio
    (sharpe ratio without factoring in the risk-free rate)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    return returns.mean() / returns.std()


def max_drawdown(prices):
    """Calculates the maximum drawdown"""
    prices = _utils._prepare_prices(prices)
    return (prices / prices.expanding(min_periods=0).max()).min() - 1


def to_drawdown_series(returns):
    """Convert returns series to drawdown series"""
    prices = _utils._prepare_prices(returns)
    dd = prices / _np.maximum.accumulate(prices) - 1.0
    return dd.replace([_np.inf, -_np.inf, -0], 0)


def drawdown_details(drawdown):
    """
    Calculates drawdown details, including start/end/valley dates,
    duration, max drawdown and max dd for 99% of the dd period
    for every drawdown period
    """

    def _drawdown_details(drawdown):
        # mark no drawdown
        no_dd = drawdown == 0

        # extract dd start dates, first date of the drawdown
        starts = ~no_dd & no_dd.shift(1)
        starts = list(starts[starts.values].index)

        # extract end dates, last date of the drawdown
        ends = no_dd & (~no_dd).shift(1)
        ends = ends.shift(-1, fill_value=False)
        ends = list(ends[ends.values].index)

        # no drawdown :)
        if not starts:
            return _pd.DataFrame(
                index=[],
                columns=(
                    "start",
                    "valley",
                    "end",
                    "days",
                    "max drawdown",
                    "99% max drawdown",
                ),
            )

        # drawdown series begins in a drawdown
        if ends and starts[0] > ends[0]:
            starts.insert(0, drawdown.index[0])

        # series ends in a drawdown fill with last date
        if not ends or starts[-1] > ends[-1]:
            ends.append(drawdown.index[-1])

        # build dataframe from results
        data = []
        for i, _ in enumerate(starts):
            dd = drawdown[starts[i] : ends[i]]
            clean_dd = -remove_outliers(-dd, 0.99)
            data.append(
                (
                    starts[i],
                    dd.idxmin(),
                    ends[i],
                    (ends[i] - starts[i]).days + 1,
                    dd.min() * 100,
                    clean_dd.min() * 100,
                )
            )

        df = _pd.DataFrame(
            data=data,
            columns=(
                "start",
                "valley",
                "end",
                "days",
                "max drawdown",
                "99% max drawdown",
            ),
        )
        df["days"] = df["days"].astype(int)
        df["max drawdown"] = df["max drawdown"].astype(float)
        df["99% max drawdown"] = df["99% max drawdown"].astype(float)

        df["start"] = df["start"].dt.strftime("%Y-%m-%d")
        df["end"] = df["end"].dt.strftime("%Y-%m-%d")
        df["valley"] = df["valley"].dt.strftime("%Y-%m-%d")

        return df

    if isinstance(drawdown, _pd.DataFrame):
        _dfs = {}
        for col in drawdown.columns:
            _dfs[col] = _drawdown_details(drawdown[col])
        return _pd.concat(_dfs, axis=1)

    return _drawdown_details(drawdown)


def kelly_criterion(returns, prepare_returns=True):
    """
    Calculates the recommended maximum amount of capital that
    should be allocated to the given strategy, based on the
    Kelly Criterion (http://en.wikipedia.org/wiki/Kelly_criterion)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    win_loss_ratio = payoff_ratio(returns)
    win_prob = win_rate(returns)
    lose_prob = 1 - win_prob

    return ((win_loss_ratio * win_prob) - lose_prob) / win_loss_ratio


# ==== VS. BENCHMARK ====


def r_squared(returns, benchmark, prepare_returns=True):
    """Measures the straight line fit of the equity curve"""
    # slope, intercept, r_val, p_val, std_err = _linregress(
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    _, _, r_val, _, _ = _linregress(
        returns, _utils._prepare_benchmark(benchmark, returns.index)
    )
    return r_val**2


def r2(returns, benchmark):
    """Shorthand for r_squared()"""
    return r_squared(returns, benchmark)


def information_ratio(returns, benchmark, prepare_returns=True):
    """
    Calculates the information ratio
    (basically the risk return ratio of the net profits)
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    diff_rets = returns - _utils._prepare_benchmark(benchmark, returns.index)

    return diff_rets.mean() / diff_rets.std()


def greeks(returns, benchmark, periods=float(PERIODS_PER_YEAR), prepare_returns=True):
    """Calculates alpha and beta of the portfolio"""
    # ----------------------------
    # data cleanup
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)
    # ----------------------------

    # find covariance
    matrix = _np.cov(returns, benchmark)
    beta = matrix[0, 1] / matrix[1, 1]

    # calculates measures now
    alpha = returns.mean() - beta * benchmark.mean()
    alpha = alpha * periods

    return _pd.Series(
        {
            "beta": beta,
            "alpha": alpha,
            # "vol": _np.sqrt(matrix[0, 0]) * _np.sqrt(periods)
        }
    ).fillna(0)


def rolling_greeks(returns, benchmark, periods=PERIODS_PER_YEAR, prepare_returns=True):
    """Calculates rolling alpha and beta of the portfolio"""
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    df = _pd.DataFrame(
        data={
            "returns": returns,
            "benchmark": _utils._prepare_benchmark(benchmark, returns.index),
        }
    )
    df = df.fillna(0)
    corr = df.rolling(int(periods)).corr().unstack()["returns"]["benchmark"]
    std = df.rolling(int(periods)).std()
    beta = corr * std["returns"] / std["benchmark"]

    alpha = df["returns"].mean() - beta * df["benchmark"].mean()

    # alpha = alpha * periods
    return _pd.DataFrame(index=returns.index, data={"beta": beta, "alpha": alpha})


def compare(
    returns,
    benchmark,
    aggregate=None,
    compounded=True,
    round_vals=None,
    prepare_returns=True,
):
    """
    Compare returns to benchmark on a
    day/week/month/quarter/year basis
    """
    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    benchmark = _utils._prepare_benchmark(benchmark, returns.index)

    if isinstance(returns, _pd.Series):
        data = _pd.DataFrame(
            data={
                "Benchmark": _utils.aggregate_returns(benchmark, aggregate, compounded)
                * 100,
                "Returns": _utils.aggregate_returns(returns, aggregate, compounded)
                * 100,
            }
        )

        data["Multiplier"] = data["Returns"] / data["Benchmark"]
        data["Won"] = _np.where(data["Returns"] >= data["Benchmark"], "+", "-")
    elif isinstance(returns, _pd.DataFrame):
        bench = {
            "Benchmark": _utils.aggregate_returns(benchmark, aggregate, compounded)
            * 100
        }
        strategy = {
            "Returns_"
            + str(i): _utils.aggregate_returns(returns[col], aggregate, compounded)
            * 100
            for i, col in enumerate(returns.columns)
        }
        data = _pd.DataFrame(data={**bench, **strategy})

    if round_vals is not None:
        return _np.round(data, round_vals)

    return data


def monthly_returns(returns, eoy=True, compounded=True, prepare_returns=True):
    """Calculates monthly returns"""
    if isinstance(returns, _pd.DataFrame):
        warn(
            "Pandas DataFrame was passed (Series expected). "
            "Only first column will be used."
        )
        returns = returns.copy()
        returns.columns = map(str.lower, returns.columns)
        if len(returns.columns) > 1 and "close" in returns.columns:
            returns = returns["close"]
        else:
            returns = returns[returns.columns[0]]

    if prepare_returns:
        returns = _utils._prepare_returns(returns)
    original_returns = returns.copy()

    returns = _pd.DataFrame(
        _utils.group_returns(returns, returns.index.strftime("%Y-%m-01"), compounded)
    )

    returns.columns = ["Returns"]
    returns.index = _pd.to_datetime(returns.index)

    # get returnsframe
    returns["Year"] = returns.index.strftime("%Y")
    returns["Month"] = returns.index.strftime("%b")

    # make pivot table
    returns = returns.pivot(index="Year", columns="Month", values="Returns").fillna(0)

    # handle missing months
    for month in [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]:
        if month not in returns.columns:
            returns.loc[:, month] = 0

    # order columns by month
    returns = returns[
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    ]

    if eoy:
        returns["eoy"] = _utils.group_returns(
            original_returns, original_returns.index.year, compounded=compounded
        ).values

    returns.columns = map(lambda x: str(x).upper(), returns.columns)
    returns.index.name = None

    return returns

def final_equity(equity):
    return equity[-1]

def max_equity(equity):
    return equity.max()

def max_possible_compound_return(close_prices, fee, fee_type="floating", buy_or_sell=0):
    return CompoundReturnCalculator().calculate_compound_return_from_prices_list(close_prices, fee=fee, fee_type=fee_type, buy_or_sell=buy_or_sell) * 100

def buy_and_hold_return(close_prices, fee=0, fee_type="floating"):
    initial_value = close_prices[0]
    final_value = close_prices[-1]

    if fee_type == "fixed":
        delta = (final_value - initial_value - fee)
    elif fee_type == "floating":
        delta = (final_value - (initial_value * (1 + fee)))

    return_pct = (delta / initial_value) * 100
    return return_pct



class CompoundReturnCalculator:
    # TO DO: Consider two-sided market
    def __init__(self):
        pass

    @staticmethod
    def find_rising_sequences(prices_list):
        rising_sequences = []
        current_sequence = []

        for value in prices_list:
            if not current_sequence or value > current_sequence[-1]:
                current_sequence.append(value)
            else:
                if len(current_sequence) > 1:
                    rising_sequences.append(current_sequence)
                current_sequence = [value]

        if len(current_sequence) > 1:
            rising_sequences.append(current_sequence)

        return rising_sequences

    @staticmethod
    def find_falling_sequences(prices_list, reverse=False):
        falling_sequences = []
        current_sequence = []

        for value in prices_list:
            if not current_sequence or value < current_sequence[-1]:
                current_sequence.append(value)
            else:
                if len(current_sequence) > 1:
                    if reverse:
                        current_sequence.reverse()
                    falling_sequences.append(current_sequence)
                current_sequence = [value]

        if len(current_sequence) > 1:
            if reverse:
                current_sequence.reverse()
            falling_sequences.append(current_sequence)

        return falling_sequences


    @staticmethod
    def calculate_returns(price_sequences, fee=0, fee_type="floating"):
        # fee_type: floating, fixed
        returns = []

        for sequence in price_sequences:
            initial_value = sequence[0]
            final_value = sequence[-1]

            if fee_type == "fixed":
                delta = (final_value - initial_value - fee)
            elif fee_type == "floating":
                delta = (final_value - (initial_value * (1+fee)))

            return_pct = (delta / initial_value)
            returns.append(return_pct)

        return returns

    @staticmethod
    def calculate_compound_return(returns):
        compound_return = 1.0
        for r in returns:
            compound_return *= (1.0 + r)
        return compound_return - 1.0

    def calculate_compound_return_from_prices_list(self, prices_list, log=False, fee=0, fee_type="floating", buy_or_sell=0):
        # buy_or_sell: 0 only buy, 1 only sell, 2 both
        if buy_or_sell==0:
            rising_sequences = self.find_rising_sequences(prices_list=prices_list)
            returns = self.calculate_returns(price_sequences=rising_sequences, fee=fee, fee_type=fee_type)
            price_sequences = rising_sequences
        elif buy_or_sell==1:
            falling_sequences = self.find_falling_sequences(prices_list=prices_list, reverse=True)
            returns = self.calculate_returns(price_sequences=falling_sequences, fee=fee, fee_type=fee_type)
            price_sequences = falling_sequences
        elif buy_or_sell==2:
            returns = []

            rising_sequences = self.find_rising_sequences(prices_list=prices_list)
            rising_returns = self.calculate_returns(price_sequences=rising_sequences, fee=fee, fee_type=fee_type)
            returns.extend(rising_returns)

            falling_sequences = self.find_falling_sequences(prices_list=prices_list, reverse=True)
            falling_returns = self.calculate_returns(price_sequences=falling_sequences, fee=fee, fee_type=fee_type)
            returns.extend(falling_returns)

        if log:
            for i, sequence in enumerate(price_sequences):
                print(f"Price Sequence {i + 1}: {sequence}")
                print(f"Return: {returns[i]}\n")

        compound_return = self.calculate_compound_return(returns=returns)

        return compound_return

class Trades:
    def __init__(self, orders):
        self.orders=orders
        self.buy_orders=self.get_buy_or_sell_orders(type_="Buy")
        self.sell_orders=self.get_buy_or_sell_orders(type_="Sell")

    def get_orders_based_on_types(self, type_):
        # type= 0: All, 1: Buy, 2: Sell
        if type_==0:
            return self.orders
        elif type_==1:
            return self.buy_orders
        elif type_==2:
            return self.sell_orders

    def get_buy_or_sell_orders(self, type_):
        # type= Buy Sell
        selected_orders = self.orders[self.orders["Type"] == type_]
        return selected_orders

    def number_of_trades(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return len(orders)

    def best_trade_return(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders["Return"].max() * 100

    def worst_trade_return(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders["Return"].min() * 100

    def average_trade_return(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders["Return"].mean() * 100

    def max_trade_duration(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders["Duration"].max()

    def average_trade_duration(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders["Duration"].mean()

    def net_profit(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders["Profit"].sum()

    def average_profit(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders["Profit"].mean()

    def std_profit(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders["Profit"].std()

    def gross_profit(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders[orders["Gross Profit"]>=0]["Gross Profit"].sum()

    def gross_loss(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return  orders[orders["Gross Profit"]<0]["Gross Profit"].sum()

    def sqn(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return _np.sqrt(self.number_of_trades(type_)) * (self.average_profit(type_) / (self.std_profit(type_) or np.nan))

    def total_paid_fees(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return orders["Paid Fee"].sum()

    def profit_factor(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return (self.gross_profit(type_) / _np.abs(self.gross_loss(type_)))

    def win_rate(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        return 100 * len(orders[orders["Profit"]>0])/len(orders[orders["Profit"] != 0])

    def get_len_orders_based_on_types(self, type_=0):
        orders = self.get_orders_based_on_types(type_)
        len_orders = len(orders)
        return len_orders
