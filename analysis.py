#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股股票分析核心逻辑模块
被 Flask (app.py) 和 Streamlit (streamlit_app.py) 共同复用。
不含任何 Web 框架代码，纯数据分析函数。
"""

import os
import time
import hashlib
import warnings
from datetime import datetime, timedelta
from functools import wraps
from http.client import RemoteDisconnected

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# 回测参数
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.00025
STAMP_TAX_RATE = 0.001
MIN_COMMISSION = 5.0

# 参与相关性分析和建模的指标列
INDICATOR_COLS = [
    'MA5', 'MA10', 'MA20', 'MA60',
    'DIF', 'DEA', 'MACD',
    'RSI6', 'RSI14',
    'K', 'D', 'J',
    'BOLL_MID', 'BOLL_UP', 'BOLL_LOW', 'BOLL_PCTB',
    'VOL_MA5', 'VOL_RATE',
    'OBV', 'WR', 'CCI',
    'CLOSE_MA20_RATIO'
]


# ============================================================
# 磁盘缓存装饰器
# ============================================================
def cache_pickle(expire_seconds=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = hashlib.md5(
                f"{func.__name__}|{args}|{sorted(kwargs.items())}".encode()
            ).hexdigest()
            cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
            if os.path.exists(cache_file):
                age = time.time() - os.path.getmtime(cache_file)
                if age < expire_seconds:
                    try:
                        return pd.read_pickle(cache_file)
                    except Exception:
                        pass
            result = func(*args, **kwargs)
            try:
                result.to_pickle(cache_file)
            except Exception:
                pass
            return result
        return wrapper
    return decorator


# ============================================================
# 1. 数据获取（多数据源自动降级机制）
# ============================================================
# 数据源优先级列表：按顺序依次尝试，第一个成功的即使用
#   1. AkShare-东方财富  —— 国内首选，前复权质量最好
#   2. AkShare-新浪财经  —— 国内备用
#   3. Yahoo Finance     —— 海外终极回退，访问极稳定
# ============================================================

def _set_default_timeout(timeout=30):
    """
    全局设置 requests 默认超时时间。
    akshare/yfinance 内部调用 requests 时通常不传 timeout，
    在海外环境容易因网络延迟导致连接中断。通过 monkeypatch 注入默认超时。
    """
    import requests
    if not hasattr(requests, '_original_get'):
        requests._original_get = requests.get
        requests._original_session_get = requests.Session.get

    def _get_with_timeout(url, **kwargs):
        kwargs.setdefault('timeout', timeout)
        return requests._original_get(url, **kwargs)

    def _session_get_with_timeout(self, url, **kwargs):
        kwargs.setdefault('timeout', timeout)
        return requests._original_session_get(self, url, **kwargs)

    requests.get = _get_with_timeout
    requests.Session.get = _session_get_with_timeout


def _setup_requests_session():
    """配置带重试和浏览器UA的 requests Session。"""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry_strategy = Retry(
        total=5, backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"], raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Connection': 'keep-alive',
    })
    return session


def _fetch_with_retry(fetch_func, max_retries=3, base_delay=2):
    """
    通用重试包装器：对任意数据获取函数进行重试。
    捕获网络相关异常（ConnectionError/Timeout/RemoteDisconnected等），
    非网络异常（如参数错误）直接抛出不重试。
    """
    import requests
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fetch_func()
            if result is not None and (not isinstance(result, pd.DataFrame) or len(result) > 0):
                return result
            last_error = ValueError("返回数据为空")
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ChunkedEncodingError,
                ConnectionError, TimeoutError,
                RemoteDisconnected, OSError) as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                print(f"[数据获取] 第{attempt}次失败 ({type(e).__name__})，{delay}秒后重试...")
                time.sleep(delay)
            continue
        except Exception as e:
            raise  # 非网络错误不重试
    raise ConnectionError(
        f"重试{max_retries}次后仍失败: {type(last_error).__name__}: {str(last_error)[:150]}"
    ) from last_error


def _standardize_dataframe(df, source_name):
    """
    将不同数据源返回的 DataFrame 统一为标准列名：
    date, open, high, low, close, volume, amount
    """
    col_map = {
        # 东方财富 / akshare 中文列名
        '日期': 'date', '开盘': 'open', '收盘': 'close',
        '最高': 'high', '最低': 'low', '成交量': 'volume',
        '成交额': 'amount', '涨跌幅': 'pct_change', '换手率': 'turnover',
        # 新浪 / 英文列名
        'date': 'date', 'open': 'open', 'high': 'high',
        'low': 'low', 'close': 'close', 'volume': 'volume',
        'outstanding_share': 'amount',
        # Yahoo Finance 列名
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close',
        'Dividends': 'dividends', 'Stock Splits': 'splits',
    }
    df = df.rename(columns=col_map)

    # 处理 Yahoo Finance 的 MultiIndex columns（yfinance 返回的列可能是多级索引）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=col_map)

    # 确保 date 列存在且格式统一
    if 'date' not in df.columns and df.index.name in ('Date', 'date', None):
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: 'date'})

    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # 确保必要列存在
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            df[col] = np.nan

    # 统一数值类型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    keep = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']
    df = df[[c for c in keep if c in df.columns]]
    df = df.sort_values('date').drop_duplicates(subset='date').reset_index(drop=True)
    df = df.ffill().bfill()

    print(f"[数据标准化] 数据源={source_name}, 获得{len(df)}条有效数据")
    return df


# ============================================================
# 数据源 1: AkShare - 东方财富（国内首选，前复权质量最好）
# ============================================================
def _fetch_akshare_eastmoney(code, start_date, end_date):
    """通过 akshare 东方财富接口获取前复权日线数据。"""
    import akshare as ak
    s = start_date.replace('-', '')
    e = end_date.replace('-', '')
    df = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=s, end_date=e, adjust="qfq"
    )
    return _standardize_dataframe(df, "akshare-eastmoney")


# ============================================================
# 数据源 2: AkShare - 新浪财经（国内备用）
# ============================================================
def _fetch_akshare_sina(code, start_date, end_date):
    """通过 akshare 新浪财经接口获取前复权日线数据。"""
    import akshare as ak
    prefix = 'sh' if code.startswith('6') else 'sz'
    symbol = f"{prefix}{code}"
    df = ak.stock_zh_a_daily(
        symbol=symbol, start_date=start_date, end_date=end_date, adjust="qfq"
    )
    return _standardize_dataframe(df, "akshare-sina")


# ============================================================
# 数据源 3: Yahoo Finance（海外终极回退，访问极稳定）
# ============================================================
def _to_yfinance_ticker(code):
    """
    将A股代码转换为 Yahoo Finance ticker 格式：
      6xxxxx (沪市) → 600519.SS
      0xxxxx/3xxxxx (深市) → 000001.SZ
      4xxxxx/8xxxxx (北交所) → 430047.BJ
    """
    if code.startswith('6'):
        return f"{code}.SS"
    elif code.startswith(('0', '3')):
        return f"{code}.SZ"
    elif code.startswith(('4', '8')):
        return f"{code}.BJ"
    else:
        return f"{code}.SS"  # 默认沪市


def _fetch_yfinance(code, start_date, end_date):
    """
    通过 yfinance 获取A股前复权日线数据。
    auto_adjust=True 时 OHLC 已按分红拆股调整（等价于前复权）。
    注意：yfinance 的 end_date 是排他的，需加一天。
    """
    import yfinance as yf
    ticker = _to_yfinance_ticker(code)
    # end_date 排他，加1天确保包含结束日
    end_inclusive = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')

    df = yf.download(
        ticker, start=start_date, end=end_inclusive,
        auto_adjust=True, progress=False, threads=False
    )
    if df is None or len(df) == 0:
        raise ValueError(f"Yahoo Finance 未找到 {ticker} 的数据")

    # yfinance 返回以日期为索引的 DataFrame
    df = df.reset_index()
    # 处理可能的 MultiIndex 列
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                             'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    return _standardize_dataframe(df, "yfinance")


# ============================================================
# 数据源优先级配置
# ============================================================
# 每个条目: (名称, 获取函数, 重试次数, 基础重试间隔秒)
DATA_SOURCES = [
    ("AkShare-东方财富", _fetch_akshare_eastmoney, 3, 2),
    ("AkShare-新浪财经", _fetch_akshare_sina, 2, 2),
    ("Yahoo Finance", _fetch_yfinance, 2, 3),
]


@cache_pickle(expire_seconds=86400)
def fetch_stock_data(code, start_date, end_date):
    """
    获取A股前复权日线数据（多数据源自动降级）。

    按 DATA_SOURCES 优先级依次尝试，每个数据源内部带重试，
    第一个成功返回有效数据的即使用。全部失败则抛出汇总错误。

    参数:
      code: 6位A股代码，如 '600519'
      start_date: 'YYYY-MM-DD'
      end_date: 'YYYY-MM-DD'
    返回:
      DataFrame [date, open, high, low, close, volume, amount, pct_change]
    """
    # 设置全局网络配置
    _set_default_timeout(timeout=30)
    _setup_requests_session()

    errors = []
    for source_name, fetch_func, max_retries, base_delay in DATA_SOURCES:
        try:
            print(f"\n[数据源] 尝试 {source_name} ...")
            df = _fetch_with_retry(
                lambda: fetch_func(code, start_date, end_date),
                max_retries=max_retries, base_delay=base_delay
            )
            if df is not None and len(df) > 0:
                print(f"[数据源] ✅ {source_name} 获取成功，共 {len(df)} 条数据")
                return df
            else:
                errors.append(f"{source_name}: 返回空数据")
        except Exception as e:
            err_msg = f"{source_name}: {type(e).__name__}: {str(e)[:120]}"
            errors.append(err_msg)
            print(f"[数据源] ❌ {err_msg}")
            continue

    # 所有数据源均失败
    error_summary = "\n  ".join(errors)
    raise ConnectionError(
        f"股票 {code} 在 {start_date}~{end_date} 的数据获取失败，所有数据源均不可用：\n  {error_summary}\n"
        f"建议: 1) 检查网络连接 2) 稍后重试 3) 确认股票代码正确 4) 扩大日期范围"
    )


def fetch_recent_data(code, days=150):
    """获取最近一段交易日数据（用于实时信号），缓存2小时。"""
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=days * 2)).strftime('%Y-%m-%d')

    @cache_pickle(expire_seconds=7200)
    def _fetch(c, s, e):
        return fetch_stock_data(c, s, e)
    return _fetch(code, start, end)

# ============================================================
# 2. 技术指标计算
# ============================================================
def calc_indicators(df):
    """计算23个技术指标，返回追加指标列的DataFrame（去除前60行预热期）。"""
    d = df.copy()
    c = d['close']
    h = d['high']
    l = d['low']
    v = d['volume']

    for w in [5, 10, 20, 60]:
        d[f'MA{w}'] = c.rolling(w).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d['DIF'] = ema12 - ema26
    d['DEA'] = d['DIF'].ewm(span=9, adjust=False).mean()
    d['MACD'] = 2 * (d['DIF'] - d['DEA'])

    for period in [6, 14]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        d[f'RSI{period}'] = 100 - 100 / (1 + rs)

    n = 9
    low_n = l.rolling(n).min()
    high_n = h.rolling(n).max()
    rsv = (c - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    d['K'] = rsv.ewm(com=2, adjust=False).mean()
    d['D'] = d['K'].ewm(com=2, adjust=False).mean()
    d['J'] = 3 * d['K'] - 2 * d['D']

    d['BOLL_MID'] = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    d['BOLL_UP'] = d['BOLL_MID'] + 2 * std20
    d['BOLL_LOW'] = d['BOLL_MID'] - 2 * std20
    d['BOLL_PCTB'] = (c - d['BOLL_LOW']) / (d['BOLL_UP'] - d['BOLL_LOW']).replace(0, np.nan)

    d['VOL_MA5'] = v.rolling(5).mean()
    d['VOL_RATE'] = (v - v.shift(1)) / v.shift(1).replace(0, np.nan) * 100

    direction = np.sign(c.diff()).fillna(0)
    d['OBV'] = (direction * v).cumsum()

    hh14 = h.rolling(14).max()
    ll14 = l.rolling(14).min()
    d['WR'] = (hh14 - c) / (hh14 - ll14).replace(0, np.nan) * -100

    tp = (h + l + c) / 3
    ma_tp = tp.rolling(14).mean()
    md = tp.rolling(14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d['CCI'] = (tp - ma_tp) / (0.015 * md.replace(0, np.nan))

    d['CLOSE_MA20_RATIO'] = (c - d['MA20']) / d['MA20'].replace(0, np.nan) * 100

    d = d.iloc[60:].reset_index(drop=True)
    d = d.ffill().bfill()
    return d


# ============================================================
# 3. 最优买卖点（贪心峰谷法）
# ============================================================
def find_optimal_trades(df, min_profit_pct=1.0):
    closes = df['close'].values
    dates = df['date'].values
    n = len(closes)
    if n < 2:
        return [], 0.0, 0.0

    trades = []
    i = 0
    while i < n - 1:
        while i < n - 1 and closes[i + 1] <= closes[i]:
            i += 1
        if i >= n - 1:
            break
        buy_idx = i
        buy_price = closes[buy_idx]
        while i < n - 1 and closes[i + 1] >= closes[i]:
            i += 1
        sell_idx = i
        sell_price = closes[sell_idx]
        profit = (sell_price - buy_price) / buy_price * 100
        if profit >= min_profit_pct:
            trades.append({
                'buy_date': str(dates[buy_idx]),
                'buy_price': round(float(buy_price), 2),
                'sell_date': str(dates[sell_idx]),
                'sell_price': round(float(sell_price), 2),
                'profit_rate': round(float(profit), 2),
                'holding_days': int(sell_idx - buy_idx)
            })

    total = 1.0
    for t in trades:
        total *= (1 + t['profit_rate'] / 100)
    total_return = round((total - 1) * 100, 2)
    buy_hold = round((closes[-1] - closes[0]) / closes[0] * 100, 2)
    return trades, total_return, buy_hold


# ============================================================
# 4. 相关性分析
# ============================================================
def analyze_correlation(df, trades):
    n = len(df)
    buy_dates = set(t['buy_date'] for t in trades)
    sell_dates = set(t['sell_date'] for t in trades)
    buy_labels = np.array([1 if d in buy_dates else 0 for d in df['date']])
    sell_labels = np.array([1 if d in sell_dates else 0 for d in df['date']])

    def _corr_for_labels(labels):
        results = []
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        for col in INDICATOR_COLS:
            vals = df[col].values.astype(float)
            if np.isnan(vals).all() or np.nanstd(vals) == 0:
                continue
            valid = ~np.isnan(vals)
            if valid.sum() < 10:
                continue
            r, p = stats.pointbiserialr(labels[valid], vals[valid])
            if np.isnan(r) or np.isnan(p):
                continue
            mean_at = float(np.nanmean(vals[labels == 1])) if n_pos > 0 else 0
            mean_other = float(np.nanmean(vals[labels == 0])) if n_neg > 0 else 0
            direction = 'high' if mean_at > mean_other else 'low'
            results.append({
                'indicator': col,
                'corr': round(float(r), 4),
                'abs_corr': round(abs(float(r)), 4),
                'pvalue': round(float(p), 6),
                'significant': bool(p < 0.05),
                'mean_at_event': round(mean_at, 4),
                'mean_other': round(mean_other, 4),
                'direction': direction
            })
        results.sort(key=lambda x: x['abs_corr'], reverse=True)
        return results

    buy_corr = _corr_for_labels(buy_labels)
    sell_corr = _corr_for_labels(sell_labels)

    def _profile(event_dates):
        event_idx = [i for i, d in enumerate(df['date']) if d in event_dates]
        profile = {}
        offsets = list(range(-3, 4))
        for col in INDICATOR_COLS:
            vals = df[col].values.astype(float)
            series = []
            for off in offsets:
                gathered = []
                for ei in event_idx:
                    j = ei + off
                    if 0 <= j < n and not np.isnan(vals[j]):
                        gathered.append(vals[j])
                series.append(round(float(np.mean(gathered)), 4) if gathered else None)
            profile[col] = {'offsets': offsets, 'values': series}
        return profile

    buy_profiles = _profile(buy_dates)
    sell_profiles = _profile(sell_dates)
    return buy_corr, sell_corr, buy_profiles, sell_profiles


# ============================================================
# 5. 数学模型（阈值加权打分系统）
# ============================================================
def build_model(df, buy_corr, sell_corr, top_n=6):
    def _make_rules(corr_list, event_type):
        rules = []
        for item in corr_list[:top_n]:
            col = item['indicator']
            weight = item['abs_corr']
            direction = item['direction']
            mean_at = item['mean_at_event']
            mean_ot = item['mean_other']
            threshold = (mean_at + mean_ot) / 2
            std_val = float(df[col].std())
            if std_val == 0 or np.isnan(std_val):
                std_val = 1.0
            rules.append({
                'indicator': col,
                'weight': round(weight, 4),
                'direction': direction,
                'threshold': round(float(threshold), 4),
                'std': round(std_val, 4),
                'mean_at_event': mean_at,
                'mean_other': mean_ot,
                'event_type': event_type
            })
        return rules

    buy_rules = _make_rules(buy_corr, 'buy')
    sell_rules = _make_rules(sell_corr, 'sell')

    model = {
        'type': '阈值加权打分系统 (Threshold-Weighted Scoring)',
        'description': (
            '选取与最优买卖点相关性最高的指标，以事件点/非事件点均值中点为阈值，'
            '以指标偏离度（标准化）×相关系数为权重，综合打分映射为买卖概率。'
        ),
        'buy_rules': buy_rules,
        'sell_rules': sell_rules,
        'buy_threshold_prob': 0.60,
        'sell_threshold_prob': 0.60,
        'formula': (
            'score_buy = Σ [ weight_i × sigmoid( (value_i - threshold_i) / std_i × sign_i ) ]\n'
            'prob_buy = score_buy / Σ weight_i\n'
            '其中 sign_i = +1 (direction=high), -1 (direction=low)'
        )
    }
    return model


def compute_model_signals(df, model):
    buy_rules = model['buy_rules']
    sell_rules = model['sell_rules']
    buy_w_sum = sum(r['weight'] for r in buy_rules) or 1
    sell_w_sum = sum(r['weight'] for r in sell_rules) or 1
    n = len(df)

    def _calc_prob(rules, w_sum):
        probs = np.zeros(n)
        for r in rules:
            col = r['indicator']
            vals = df[col].values.astype(float)
            sign = 1 if r['direction'] == 'high' else -1
            threshold = r['threshold']
            std = r['std'] or 1.0
            weight = r['weight']
            z = sign * (vals - threshold) / std
            sig = 1 / (1 + np.exp(-np.clip(z, -10, 10)))
            probs += weight * sig
        return probs / w_sum

    buy_probs = _calc_prob(buy_rules, buy_w_sum)
    sell_probs = _calc_prob(sell_rules, sell_w_sum)
    return buy_probs, sell_probs


# ============================================================
# 6. 历史回测
# ============================================================
def backtest(df, model, buy_probs, sell_probs):
    buy_th = model['buy_threshold_prob']
    sell_th = model['sell_threshold_prob']
    n = len(df)
    cash = INITIAL_CAPITAL
    shares = 0
    equity = np.zeros(n)
    trades = []
    position = None

    for i in range(n):
        if i < n - 1:
            exec_price = float(df['open'].iloc[i + 1])
            exec_date = str(df['date'].iloc[i + 1])

            if position is None and buy_probs[i] >= buy_th:
                max_cost = cash / (1 + COMMISSION_RATE)
                buy_shares = int(max_cost / exec_price / 100) * 100
                if buy_shares >= 100:
                    cost = buy_shares * exec_price
                    commission = max(cost * COMMISSION_RATE, MIN_COMMISSION)
                    total_cost = cost + commission
                    if total_cost <= cash:
                        cash -= total_cost
                        shares = buy_shares
                        position = {
                            'buy_date': exec_date,
                            'buy_price': round(exec_price, 2),
                            'shares': buy_shares,
                            'buy_cost': round(total_cost, 2),
                            'buy_prob': round(float(buy_probs[i]), 4)
                        }

            elif position is not None and sell_probs[i] >= sell_th:
                revenue = shares * exec_price
                commission = max(revenue * COMMISSION_RATE, MIN_COMMISSION)
                stamp_tax = revenue * STAMP_TAX_RATE
                net_revenue = revenue - commission - stamp_tax
                cash += net_revenue
                profit = net_revenue - position['buy_cost']
                profit_rate = profit / position['buy_cost'] * 100
                trades.append({
                    'buy_date': position['buy_date'],
                    'buy_price': position['buy_price'],
                    'sell_date': exec_date,
                    'sell_price': round(exec_price, 2),
                    'shares': position['shares'],
                    'profit': round(profit, 2),
                    'profit_rate': round(profit_rate, 2),
                    'buy_prob': position['buy_prob'],
                    'sell_prob': round(float(sell_probs[i]), 4)
                })
                shares = 0
                position = None

        equity[i] = cash + shares * float(df['close'].iloc[i])

    if position is not None and shares > 0:
        last_price = float(df['close'].iloc[-1])
        revenue = shares * last_price
        commission = max(revenue * COMMISSION_RATE, MIN_COMMISSION)
        stamp_tax = revenue * STAMP_TAX_RATE
        net_revenue = revenue - commission - stamp_tax
        cash += net_revenue
        profit = net_revenue - position['buy_cost']
        profit_rate = profit / position['buy_cost'] * 100
        trades.append({
            'buy_date': position['buy_date'],
            'buy_price': position['buy_price'],
            'sell_date': str(df['date'].iloc[-1]) + '(期末平仓)',
            'sell_price': round(last_price, 2),
            'shares': position['shares'],
            'profit': round(profit, 2),
            'profit_rate': round(profit_rate, 2),
            'buy_prob': position['buy_prob'],
            'sell_prob': 1.0
        })
        equity[-1] = cash
        shares = 0
        position = None

    equity_series = pd.Series(equity)
    total_return = (equity[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    days = len(df)
    years = days / 252
    annual_return = ((equity[-1] / INITIAL_CAPITAL) ** (1 / years) - 1) * 100 if years > 0 else 0
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak * 100
    max_drawdown = float(drawdown.min())
    win_trades = [t for t in trades if t['profit'] > 0]
    win_rate = len(win_trades) / len(trades) * 100 if trades else 0
    avg_win = np.mean([t['profit'] for t in win_trades]) if win_trades else 0
    loss_trades = [t for t in trades if t['profit'] <= 0]
    avg_loss = abs(np.mean([t['profit'] for t in loss_trades])) if loss_trades else 0
    profit_loss_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else float('inf')
    bh_return = (float(df['close'].iloc[-1]) - float(df['close'].iloc[0])) / float(df['close'].iloc[0]) * 100

    metrics = {
        'initial_capital': INITIAL_CAPITAL,
        'final_equity': round(float(equity[-1]), 2),
        'total_return': round(float(total_return), 2),
        'annual_return': round(float(annual_return), 2),
        'max_drawdown': round(float(max_drawdown), 2),
        'win_rate': round(float(win_rate), 2),
        'profit_loss_ratio': profit_loss_ratio if profit_loss_ratio != float('inf') else '∞',
        'total_trades': len(trades),
        'win_trades': len(win_trades),
        'loss_trades': len(loss_trades),
        'buy_hold_return': round(float(bh_return), 2),
        'excess_return': round(float(total_return - bh_return), 2)
    }

    step = max(1, n // 500)
    equity_curve = [
        {'date': str(df['date'].iloc[i]), 'equity': round(float(equity[i]), 2)}
        for i in range(0, n, step)
    ]
    if equity_curve[-1]['date'] != str(df['date'].iloc[-1]):
        equity_curve.append({'date': str(df['date'].iloc[-1]), 'equity': round(float(equity[-1]), 2)})

    return equity_curve, trades, metrics


# ============================================================
# 7. 预测准确率
# ============================================================
def evaluate_prediction_accuracy(df, buy_probs, sell_probs, horizon=5):
    n = len(df)
    closes = df['close'].values
    buy_th = 0.60
    sell_th = 0.60
    buy_signals = []
    sell_signals = []

    for i in range(n - horizon):
        if buy_probs[i] >= buy_th:
            future_max = float(np.max(closes[i+1:i+1+horizon]))
            current = float(closes[i])
            rose = future_max > current * 1.01
            buy_signals.append({
                'date': str(df['date'].iloc[i]),
                'prob': round(float(buy_probs[i]), 4),
                'future_return': round((future_max - current) / current * 100, 2),
                'correct': rose
            })
        if sell_probs[i] >= sell_th:
            future_min = float(np.min(closes[i+1:i+1+horizon]))
            current = float(closes[i])
            fell = future_min < current * 0.99
            sell_signals.append({
                'date': str(df['date'].iloc[i]),
                'prob': round(float(sell_probs[i]), 4),
                'future_return': round((future_min - current) / current * 100, 2),
                'correct': fell
            })

    buy_acc = sum(1 for s in buy_signals if s['correct']) / len(buy_signals) * 100 if buy_signals else 0
    sell_acc = sum(1 for s in sell_signals if s['correct']) / len(sell_signals) * 100 if sell_signals else 0

    return {
        'horizon': horizon,
        'buy_signal_count': len(buy_signals),
        'buy_accuracy': round(buy_acc, 2),
        'sell_signal_count': len(sell_signals),
        'sell_accuracy': round(sell_acc, 2),
        'buy_signals_sample': buy_signals[:20],
        'sell_signals_sample': sell_signals[:20]
    }


# ============================================================
# 8. 实时信号
# ============================================================
def generate_realtime_signal(code):
    df_raw = fetch_recent_data(code, days=150)
    if len(df_raw) < 70:
        raise ValueError(f"股票 {code} 近期数据不足，无法计算实时信号")

    df = calc_indicators(df_raw)
    trades, _, _ = find_optimal_trades(df, min_profit_pct=1.0)
    if len(trades) < 2:
        trades, _, _ = find_optimal_trades(df, min_profit_pct=0.5)

    buy_corr, sell_corr, _, _ = analyze_correlation(df, trades)
    model = build_model(df, buy_corr, sell_corr, top_n=6)
    buy_probs, sell_probs = compute_model_signals(df, model)

    latest = df.iloc[-1]
    latest_buy_prob = float(buy_probs[-1])
    latest_sell_prob = float(sell_probs[-1])

    if latest_buy_prob >= model['buy_threshold_prob'] and latest_buy_prob > latest_sell_prob:
        signal = '买入'
        signal_strength = latest_buy_prob
    elif latest_sell_prob >= model['sell_threshold_prob'] and latest_sell_prob > latest_buy_prob:
        signal = '卖出'
        signal_strength = latest_sell_prob
    else:
        signal = '持有'
        signal_strength = max(latest_buy_prob, latest_sell_prob)

    key_indicators = []
    active_rules = model['buy_rules'] if signal == '买入' else (model['sell_rules'] if signal == '卖出' else model['buy_rules'][:3] + model['sell_rules'][:3])
    for r in active_rules[:6]:
        col = r['indicator']
        val = float(latest[col])
        sign = 1 if r['direction'] == 'high' else -1
        triggered = sign * (val - r['threshold']) > 0
        key_indicators.append({
            'indicator': col,
            'value': round(val, 4),
            'threshold': r['threshold'],
            'direction': r['direction'],
            'weight': r['weight'],
            'triggered': bool(triggered)
        })

    return {
        'code': code,
        'date': str(latest['date']),
        'close': round(float(latest['close']), 2),
        'open': round(float(latest['open']), 2),
        'high': round(float(latest['high']), 2),
        'low': round(float(latest['low']), 2),
        'volume': int(latest['volume']),
        'signal': signal,
        'signal_strength': round(signal_strength, 4),
        'buy_probability': round(latest_buy_prob, 4),
        'sell_probability': round(latest_sell_prob, 4),
        'key_indicators': key_indicators,
        'model_threshold_buy': model['buy_threshold_prob'],
        'model_threshold_sell': model['sell_threshold_prob']
    }


# ============================================================
# 9. 完整分析流水线（一键调用）
# ============================================================
def run_full_analysis(code, start_date, end_date):
    """
    执行完整分析流水线，返回所有结果。
    被 Flask 和 Streamlit 共同调用。
    """
    from datetime import datetime, timedelta
    sd = datetime.strptime(start_date, '%Y-%m-%d')
    fetch_start = (sd - timedelta(days=120)).strftime('%Y-%m-%d')
    df_raw = fetch_stock_data(code, fetch_start, end_date)
    if len(df_raw) < 70:
        raise ValueError(f'数据量不足（仅{len(df_raw)}条），请扩大日期范围')

    df_full = calc_indicators(df_raw)
    df = df_full[df_full['date'] >= start_date].reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(f'选定时间段内有效数据不足（仅{len(df)}条），请扩大范围')

    trades, total_return, buy_hold_return = find_optimal_trades(df, min_profit_pct=1.0)
    buy_corr, sell_corr, buy_profiles, sell_profiles = analyze_correlation(df, trades)
    model = build_model(df, buy_corr, sell_corr, top_n=6)
    buy_probs, sell_probs = compute_model_signals(df, model)
    equity_curve, backtest_trades, metrics = backtest(df, model, buy_probs, sell_probs)
    accuracy = evaluate_prediction_accuracy(df, buy_probs, sell_probs, horizon=5)

    # K线数据
    step = max(1, len(df) // 800)
    kline_data = []
    for i in range(0, len(df), step):
        row = df.iloc[i]
        kline_data.append({
            'date': str(row['date']),
            'open': round(float(row['open']), 2),
            'close': round(float(row['close']), 2),
            'low': round(float(row['low']), 2),
            'high': round(float(row['high']), 2),
            'volume': int(row['volume'])
        })

    # 买卖点标记
    buy_markers = [{'date': t['buy_date'], 'price': t['buy_price']} for t in trades]
    sell_markers = [{'date': t['sell_date'], 'price': t['sell_price']} for t in trades]

    # 概率曲线
    prob_curve = []
    for i in range(0, len(df), step):
        prob_curve.append({
            'date': str(df['date'].iloc[i]),
            'buy_prob': round(float(buy_probs[i]), 4),
            'sell_prob': round(float(sell_probs[i]), 4)
        })

    return {
        'code': code,
        'start_date': start_date,
        'end_date': end_date,
        'data_points': len(df),
        'optimal_trades': trades,
        'optimal_total_return': total_return,
        'optimal_buy_hold_return': buy_hold_return,
        'trade_count': len(trades),
        'kline_data': kline_data,
        'buy_markers': buy_markers,
        'sell_markers': sell_markers,
        'buy_correlation': buy_corr,
        'sell_correlation': sell_corr,
        'buy_profiles': buy_profiles,
        'sell_profiles': sell_profiles,
        'model': model,
        'probability_curve': prob_curve,
        'equity_curve': equity_curve,
        'backtest_trades': backtest_trades,
        'metrics': metrics,
        'accuracy': accuracy
    }
