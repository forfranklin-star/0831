#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
波段顶/底识别与参数优化模块
============================
多种算法识别波段顶和底，通过参数网格搜索+回测验证找到最优参数组合。

支持算法：
  1. 摆动高低点（Swing High/Low）— 左右N根K线极值
  2. 分形（Fractal）— 5根K线中间极值
  3. ZigZag — 价格回撤百分比转折点
  4. MACD背离 — 价格创新高/低但MACD不创新高/低
  5. RSI超买超卖+反转确认
  6. KDJ超买超卖+死叉/金叉
  7. 布林带触碰+反转
  8. 成交量确认（顶部放量滞涨/底部放量抗跌）
  9. 综合打分系统（多算法加权投票）

参数优化：
  - 摆动点左右K线数：2,3,5
  - ZigZag回撤：5%,8%,10%,15%
  - RSI阈值：70/30, 75/25
  - 综合打分阈值：3,4,5
  - 回测持有期：3,5,10日
"""
import numpy as np
import pandas as pd
from itertools import product


# ============================================================
# 算法1：摆动高低点（Swing High/Low）
# ============================================================

def detect_swing_points(df, left=3, right=3):
    """
    检测摆动高点和低点。
    摆动高点：中间K线高点 > 左右各left/right根K线的高点
    摆动低点：中间K线低点 < 左右各left/right根K线的低点

    返回: (swing_highs_indices, swing_lows_indices)
    """
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)
    swing_highs = []
    swing_lows = []

    for i in range(left, n - right):
        # 摆动高点
        is_high = True
        for j in range(1, left + 1):
            if highs[i] <= highs[i - j]:
                is_high = False
                break
        if is_high:
            for j in range(1, right + 1):
                if highs[i] <= highs[i + j]:
                    is_high = False
                    break
        if is_high:
            swing_highs.append(i)

        # 摆动低点
        is_low = True
        for j in range(1, left + 1):
            if lows[i] >= lows[i - j]:
                is_low = False
                break
        if is_low:
            for j in range(1, right + 1):
                if lows[i] >= lows[i + j]:
                    is_low = False
                    break
        if is_low:
            swing_lows.append(i)

    return swing_highs, swing_lows


# ============================================================
# 算法2：分形（Fractal）
# ============================================================

def detect_fractals(df):
    """
    检测分形顶和分形底（5根K线）。
    分形顶：第3根K线高点 > 第1、2、4、5根K线高点
    分形底：第3根K线低点 < 第1、2、4、5根K线低点
    """
    return detect_swing_points(df, left=2, right=2)


# ============================================================
# 算法3：ZigZag转折点
# ============================================================

def detect_zigzag(df, retracement_pct=0.08):
    """
    ZigZag算法识别主要转折点。
    从一个极值点开始，价格反向变动超过retracement_pct则确认转折点。

    返回: (zigzag_highs_indices, zigzag_lows_indices)
    """
    closes = df['close'].values
    n = len(df)
    if n < 5:
        return [], []

    zigzag_highs = []
    zigzag_lows = []

    # 找到第一个极值点
    trend = 0  # 1=上升, -1=下降
    extreme_idx = 0
    extreme_price = closes[0]

    for i in range(1, n):
        if trend == 0:
            # 初始趋势判断
            if closes[i] > extreme_price * (1 + retracement_pct * 0.5):
                trend = 1
                extreme_idx = i
                extreme_price = closes[i]
            elif closes[i] < extreme_price * (1 - retracement_pct * 0.5):
                trend = -1
                extreme_idx = i
                extreme_price = closes[i]
            continue

        if trend == 1:
            # 上升趋势中，更新最高点
            if closes[i] > extreme_price:
                extreme_idx = i
                extreme_price = closes[i]
            # 回撤超过阈值，确认顶部
            elif closes[i] < extreme_price * (1 - retracement_pct):
                zigzag_highs.append(extreme_idx)
                trend = -1
                extreme_idx = i
                extreme_price = closes[i]
        else:
            # 下降趋势中，更新最低点
            if closes[i] < extreme_price:
                extreme_idx = i
                extreme_price = closes[i]
            # 反弹超过阈值，确认底部
            elif closes[i] > extreme_price * (1 + retracement_pct):
                zigzag_lows.append(extreme_idx)
                trend = 1
                extreme_idx = i
                extreme_price = closes[i]

    return zigzag_highs, zigzag_lows


# ============================================================
# 算法4：MACD背离
# ============================================================

def detect_macd_divergence(df, lookback=20):
    """
    检测MACD顶背离和底背离。
    顶背离：价格创新高但MACD_DIF不创新高
    底背离：价格创新低但MACD_DIF不创新低
    """
    if 'MACD_DIF' not in df.columns:
        return [], []

    closes = df['close'].values
    dif = df['MACD_DIF'].values
    n = len(df)
    bearish_div = []  # 顶背离
    bullish_div = []  # 底背离

    for i in range(lookback, n):
        # 近期价格极值
        recent_high_idx = i - lookback + np.argmax(closes[i - lookback:i])
        recent_low_idx = i - lookback + np.argmin(closes[i - lookback:i])

        # 当前是否创新高
        if closes[i] >= closes[recent_high_idx] * 0.995:
            # 价格创新高，但DIF没有创新高 → 顶背离
            if dif[i] < dif[recent_high_idx] * 0.98:
                bearish_div.append(i)

        # 当前是否创新低
        if closes[i] <= closes[recent_low_idx] * 1.005:
            # 价格创新低，但DIF没有创新低 → 底背离
            if dif[i] > dif[recent_low_idx] * 1.02:
                bullish_div.append(i)

    return bearish_div, bullish_div


# ============================================================
# 算法5：RSI超买超卖+反转
# ============================================================

def detect_rsi_reversal(df, overbought=70, oversold=30):
    """
    检测RSI超买后回落（顶部信号）和超卖后回升（底部信号）。
    """
    if 'RSI_14' not in df.columns:
        return [], []

    rsi = df['RSI_14'].values
    n = len(df)
    tops = []
    bottoms = []

    for i in range(2, n):
        # 顶部：RSI从超买区回落
        if rsi[i - 1] > overbought and rsi[i] < rsi[i - 1]:
            tops.append(i)
        # 底部：RSI从超卖区回升
        if rsi[i - 1] < oversold and rsi[i] > rsi[i - 1]:
            bottoms.append(i)

    return tops, bottoms


# ============================================================
# 算法6：KDJ超买超卖+死叉/金叉
# ============================================================

def detect_kdj_signals(df, overbought=80, oversold=20):
    """
    检测KDJ超买死叉（顶部）和超卖金叉（底部）。
    """
    if 'KDJ_K' not in df.columns or 'KDJ_D' not in df.columns:
        return [], []

    k = df['KDJ_K'].values
    d = df['KDJ_D'].values
    n = len(df)
    tops = []
    bottoms = []

    for i in range(1, n):
        # 顶部：K在超买区死叉D
        if k[i - 1] > overbought and k[i] < d[i] and k[i - 1] >= d[i - 1]:
            tops.append(i)
        # 底部：K在超卖区金叉D
        if k[i - 1] < oversold and k[i] > d[i] and k[i - 1] <= d[i - 1]:
            bottoms.append(i)

    return tops, bottoms


# ============================================================
# 算法7：布林带触碰+反转
# ============================================================

def detect_bollinger_reversal(df):
    """
    检测价格触碰布林上轨后回落（顶部）和触碰下轨后回升（底部）。
    """
    if 'BOLL_UPPER' not in df.columns or 'BOLL_LOWER' not in df.columns:
        return [], []

    close = df['close'].values
    upper = df['BOLL_UPPER'].values
    lower = df['BOLL_LOWER'].values
    n = len(df)
    tops = []
    bottoms = []

    for i in range(1, n):
        # 顶部：前一日触碰/突破上轨，今日回落
        if close[i - 1] >= upper[i - 1] * 0.99 and close[i] < close[i - 1]:
            tops.append(i)
        # 底部：前一日触碰/跌破下轨，今日回升
        if close[i - 1] <= lower[i - 1] * 1.01 and close[i] > close[i - 1]:
            bottoms.append(i)

    return tops, bottoms


# ============================================================
# 算法8：成交量确认
# ============================================================

def detect_volume_confirmation(df, vol_ratio_threshold=1.5):
    """
    顶部放量滞涨和底部放量抗跌。
    """
    if 'volume' not in df.columns:
        return [], []

    close = df['close'].values
    volume = df['volume'].values
    n = len(df)
    tops = []
    bottoms = []

    for i in range(5, n):
        avg_vol = np.mean(volume[i - 5:i])
        if avg_vol == 0:
            continue
        vol_ratio = volume[i] / avg_vol
        pct_change = (close[i] - close[i - 1]) / close[i - 1] * 100 if close[i - 1] > 0 else 0

        # 顶部：放量但涨幅小（滞涨）
        if vol_ratio > vol_ratio_threshold and 0 <= pct_change < 1.0:
            tops.append(i)
        # 底部：放量但跌幅小（抗跌）
        if vol_ratio > vol_ratio_threshold and -1.0 < pct_change <= 0:
            bottoms.append(i)

    return tops, bottoms


# ============================================================
# 综合打分系统
# ============================================================

def detect_composite_swing_points(df, params=None):
    """
    综合多种算法的加权打分系统识别波段顶/底。

    参数:
      params: dict，包含各算法权重和阈值
        - swing_left/right: 摆动点左右K线数
        - zigzag_pct: ZigZag回撤百分比
        - rsi_ob/os: RSI超买超卖阈值
        - kdj_ob/os: KDJ超买超卖阈值
        - vol_ratio: 量比阈值
        - top_threshold/bottom_threshold: 综合打分阈值
        - weights: 各算法权重

    返回: (top_indices, bottom_indices, top_scores, bottom_scores)
    """
    if params is None:
        params = {
            'swing_left': 3, 'swing_right': 3,
            'zigzag_pct': 0.08,
            'rsi_ob': 70, 'rsi_os': 30,
            'kdj_ob': 80, 'kdj_os': 20,
            'vol_ratio': 1.5,
            'top_threshold': 3, 'bottom_threshold': 3,
            'weights': {
                'swing': 2.0, 'fractal': 1.0, 'zigzag': 2.0,
                'macd_div': 1.5, 'rsi': 1.0, 'kdj': 1.0,
                'bollinger': 1.0, 'volume': 1.5,
            }
        }

    w = params['weights']
    n = len(df)
    top_scores = np.zeros(n)
    bottom_scores = np.zeros(n)

    # 1. 摆动点
    sh, sl = detect_swing_points(df, params['swing_left'], params['swing_right'])
    for i in sh:
        top_scores[i] += w['swing']
    for i in sl:
        bottom_scores[i] += w['swing']

    # 2. 分形
    fh, fl = detect_fractals(df)
    for i in fh:
        top_scores[i] += w['fractal']
    for i in fl:
        bottom_scores[i] += w['fractal']

    # 3. ZigZag
    zh, zl = detect_zigzag(df, params['zigzag_pct'])
    for i in zh:
        top_scores[i] += w['zigzag']
    for i in zl:
        bottom_scores[i] += w['zigzag']

    # 4. MACD背离
    mh, ml = detect_macd_divergence(df)
    for i in mh:
        top_scores[i] += w['macd_div']
    for i in ml:
        bottom_scores[i] += w['macd_div']

    # 5. RSI
    rh, rl = detect_rsi_reversal(df, params['rsi_ob'], params['rsi_os'])
    for i in rh:
        top_scores[i] += w['rsi']
    for i in rl:
        bottom_scores[i] += w['rsi']

    # 6. KDJ
    kh, kl = detect_kdj_signals(df, params['kdj_ob'], params['kdj_os'])
    for i in kh:
        top_scores[i] += w['kdj']
    for i in kl:
        bottom_scores[i] += w['kdj']

    # 7. 布林带
    bh, bl = detect_bollinger_reversal(df)
    for i in bh:
        top_scores[i] += w['bollinger']
    for i in bl:
        bottom_scores[i] += w['bollinger']

    # 8. 成交量
    vh, vl = detect_volume_confirmation(df, params['vol_ratio'])
    for i in vh:
        top_scores[i] += w['volume']
    for i in vl:
        bottom_scores[i] += w['volume']

    # 筛选超过阈值的点
    top_indices = [i for i in range(n) if top_scores[i] >= params['top_threshold']]
    bottom_indices = [i for i in range(n) if bottom_scores[i] >= params['bottom_threshold']]

    return top_indices, bottom_indices, top_scores, bottom_scores


# ============================================================
# 回测验证
# ============================================================

def backtest_swing_points(df, top_indices, bottom_indices, holding_periods=[3, 5, 10]):
    """
    回测波段顶/底识别的准确率。

    顶部信号后N日：下跌概率、平均跌幅、最大跌幅
    底部信号后N日：上涨概率、平均涨幅、最大涨幅
    """
    closes = df['close'].values
    n = len(df)
    results = {'tops': {}, 'bottoms': {}}

    for days in holding_periods:
        # 顶部回测
        top_rets = []
        for i in top_indices:
            if i + days < n:
                ret = (closes[i + days] - closes[i]) / closes[i] * 100
                top_rets.append(ret)

        if top_rets:
            results['tops'][f'{days}d'] = {
                'count': len(top_rets),
                'decline_rate': round(sum(1 for r in top_rets if r < 0) / len(top_rets) * 100, 1),
                'avg_return': round(np.mean(top_rets), 2),
                'avg_decline': round(np.mean([r for r in top_rets if r < 0]), 2) if any(r < 0 for r in top_rets) else 0,
                'max_decline': round(min(top_rets), 2),
                'max_rise': round(max(top_rets), 2),
            }

        # 底部回测
        bottom_rets = []
        for i in bottom_indices:
            if i + days < n:
                ret = (closes[i + days] - closes[i]) / closes[i] * 100
                bottom_rets.append(ret)

        if bottom_rets:
            results['bottoms'][f'{days}d'] = {
                'count': len(bottom_rets),
                'rise_rate': round(sum(1 for r in bottom_rets if r > 0) / len(bottom_rets) * 100, 1),
                'avg_return': round(np.mean(bottom_rets), 2),
                'avg_rise': round(np.mean([r for r in bottom_rets if r > 0]), 2) if any(r > 0 for r in bottom_rets) else 0,
                'max_rise': round(max(bottom_rets), 2),
                'max_decline': round(min(bottom_rets), 2),
            }

    return results


# ============================================================
# 参数网格搜索
# ============================================================

def grid_search_params(df, holding_days=5, max_combos=50):
    """
    参数网格搜索，找到识别率最高的参数组合。

    搜索空间：
      - swing_left/right: 2,3,5
      - zigzag_pct: 0.05, 0.08, 0.10, 0.15
      - top/bottom_threshold: 2,3,4,5
      - rsi_ob/os: (70,30), (75,25)
      - kdj_ob/os: (80,20), (85,15)

    评估指标：顶部下跌率 + 底部上涨率（加权平均）
    """
    # 参数网格
    swing_vals = [2, 3, 5]
    zigzag_vals = [0.05, 0.08, 0.10, 0.15]
    threshold_vals = [2, 3, 4, 5]
    rsi_vals = [(70, 30), (75, 25)]
    kdj_vals = [(80, 20), (85, 15)]

    # 生成所有组合（限制数量）
    all_combos = list(product(swing_vals, zigzag_vals, threshold_vals, rsi_vals, kdj_vals))
    if len(all_combos) > max_combos:
        # 均匀采样
        step = len(all_combos) // max_combos
        all_combos = all_combos[::step][:max_combos]

    results = []
    for swing_lr, zz_pct, thresh, (rsi_ob, rsi_os), (kdj_ob, kdj_os) in all_combos:
        params = {
            'swing_left': swing_lr, 'swing_right': swing_lr,
            'zigzag_pct': zz_pct,
            'rsi_ob': rsi_ob, 'rsi_os': rsi_os,
            'kdj_ob': kdj_ob, 'kdj_os': kdj_os,
            'vol_ratio': 1.5,
            'top_threshold': thresh, 'bottom_threshold': thresh,
            'weights': {
                'swing': 2.0, 'fractal': 1.0, 'zigzag': 2.0,
                'macd_div': 1.5, 'rsi': 1.0, 'kdj': 1.0,
                'bollinger': 1.0, 'volume': 1.5,
            }
        }

        try:
            tops, bottoms, _, _ = detect_composite_swing_points(df, params)
            bt = backtest_swing_points(df, tops, bottoms, [holding_days])

            top_rate = bt['tops'].get(f'{holding_days}d', {}).get('decline_rate', 0)
            bottom_rate = bt['bottoms'].get(f'{holding_days}d', {}).get('rise_rate', 0)
            top_count = bt['tops'].get(f'{holding_days}d', {}).get('count', 0)
            bottom_count = bt['bottoms'].get(f'{holding_days}d', {}).get('count', 0)

            # 综合评分：顶部下跌率和底部上涨率的加权平均，信号数量适中
            total_signals = top_count + bottom_count
            if total_signals > 0:
                # 信号数量惩罚：太多信号（>数据量10%）扣分
                signal_penalty = max(0, (total_signals / len(df) - 0.1) * 100)
                # 数量太少（<5）也扣分
                count_penalty = max(0, (5 - min(top_count, bottom_count)) * 2)
                score = (top_rate + bottom_rate) / 2 - signal_penalty - count_penalty
            else:
                score = -100

            results.append({
                'params': params,
                'top_rate': top_rate,
                'bottom_rate': bottom_rate,
                'top_count': top_count,
                'bottom_count': bottom_count,
                'score': round(score, 2),
            })
        except Exception:
            continue

    # 按评分排序
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


# ============================================================
# 主入口：完整波段分析
# ============================================================

def run_swing_analysis(df, timeframe='daily', holding_days=5):
    """
    完整的波段顶/底分析流程。

    1. 使用默认参数检测波段点
    2. 参数网格搜索找到最优参数
    3. 使用最优参数重新检测
    4. 回测验证
    5. 返回详细结果
    """
    if df is None or len(df) < 60:
        return {'error': '数据不足，至少需要60根K线'}

    df = df.reset_index(drop=True)

    # 1. 默认参数检测
    default_params = {
        'swing_left': 3, 'swing_right': 3,
        'zigzag_pct': 0.08,
        'rsi_ob': 70, 'rsi_os': 30,
        'kdj_ob': 80, 'kdj_os': 20,
        'vol_ratio': 1.5,
        'top_threshold': 3, 'bottom_threshold': 3,
        'weights': {
            'swing': 2.0, 'fractal': 1.0, 'zigzag': 2.0,
            'macd_div': 1.5, 'rsi': 1.0, 'kdj': 1.0,
            'bollinger': 1.0, 'volume': 1.5,
        }
    }

    default_tops, default_bottoms, default_top_scores, default_bottom_scores = \
        detect_composite_swing_points(df, default_params)
    default_bt = backtest_swing_points(df, default_tops, default_bottoms, [3, 5, 10])

    # 2. 参数网格搜索（限制组合数量，60分钟图数据多可以多搜一些）
    max_combos = 30 if timeframe == '60min' else 50
    grid_results = grid_search_params(df, holding_days=holding_days, max_combos=max_combos)

    # 3. 使用最优参数重新检测
    best_result = grid_results[0] if grid_results else None
    best_params = best_result['params'] if best_result else default_params

    best_tops, best_bottoms, best_top_scores, best_bottom_scores = \
        detect_composite_swing_points(df, best_params)
    best_bt = backtest_swing_points(df, best_tops, best_bottoms, [3, 5, 10, 20])

    # 4. 整理波段点详情
    def format_points(indices, scores, df):
        points = []
        for i in indices:
            points.append({
                'date': str(df.iloc[i]['date']),
                'index': i,
                'price': round(float(df.iloc[i]['close']), 2),
                'high': round(float(df.iloc[i]['high']), 2),
                'low': round(float(df.iloc[i]['low']), 2),
                'score': round(float(scores[i]), 2),
            })
        return points

    top_points = format_points(best_tops, best_top_scores, df)
    bottom_points = format_points(best_bottoms, best_bottom_scores, df)

    # 5. 各算法单独表现（用于展示）
    algo_performance = {}
    algos = {
        '摆动点(3,3)': lambda: detect_swing_points(df, 3, 3),
        '分形': lambda: detect_fractals(df),
        'ZigZag(8%)': lambda: detect_zigzag(df, 0.08),
        'MACD背离': lambda: detect_macd_divergence(df),
        'RSI反转': lambda: detect_rsi_reversal(df),
        'KDJ信号': lambda: detect_kdj_signals(df),
        '布林反转': lambda: detect_bollinger_reversal(df),
        '量价确认': lambda: detect_volume_confirmation(df),
    }

    for name, func in algos.items():
        try:
            tops, bottoms = func()
            bt = backtest_swing_points(df, tops, bottoms, [holding_days])
            algo_performance[name] = {
                'top_count': len(tops),
                'bottom_count': len(bottoms),
                'top_decline_rate': bt['tops'].get(f'{holding_days}d', {}).get('decline_rate', 0),
                'bottom_rise_rate': bt['bottoms'].get(f'{holding_days}d', {}).get('rise_rate', 0),
            }
        except Exception:
            algo_performance[name] = {'error': '计算失败'}

    # 6. 最优参数对比
    params_comparison = []
    for r in grid_results[:10]:
        p = r['params']
        params_comparison.append({
            'score': r['score'],
            'top_rate': r['top_rate'],
            'bottom_rate': r['bottom_rate'],
            'top_count': r['top_count'],
            'bottom_count': r['bottom_count'],
            'swing': p['swing_left'],
            'zigzag': f"{p['zigzag_pct']*100:.0f}%",
            'threshold': p['top_threshold'],
            'rsi': f"{p['rsi_ob']}/{p['rsi_os']}",
            'kdj': f"{p['kdj_ob']}/{p['kdj_os']}",
        })

    return {
        'timeframe': timeframe,
        'default_params': default_params,
        'best_params': best_params,
        'default_backtest': default_bt,
        'best_backtest': best_bt,
        'top_points': top_points,
        'bottom_points': bottom_points,
        'top_count': len(top_points),
        'bottom_count': len(bottom_points),
        'algo_performance': algo_performance,
        'grid_search_top10': params_comparison,
        'holding_days': holding_days,
    }
