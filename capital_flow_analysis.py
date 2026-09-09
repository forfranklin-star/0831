#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资金流分析模块
==============
功能：
  1. 检索个股资金流入/流出数据（主力、超大单、大单、中单、小单）
  2. 估算主力仓位变化和平均成本
  3. 识别主力买卖行为模式（吸筹/拉升/派发/洗盘）
  4. 资金流-价格模式分析（资金流入后价格延续概率）
  5. 持续学习：将学习结果存入learning模块
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import learning


def fetch_capital_flow_akshare(code, start_date, end_date):
    """
    通过akshare获取个股资金流数据。
    返回 DataFrame: date, main_net_inflow, super_large_net, large_net,
                     medium_net, small_net, main_net_pct, close
    """
    import akshare as ak

    # 判断市场
    if code.startswith(('60', '68', '90', '11', '13', '51', '58')):
        market = 'sh'
    else:
        market = 'sz'

    try:
        # akshare 个股资金流
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 统一列名（akshare不同版本列名可能不同）
        col_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if '日期' in col or 'date' in col_lower:
                col_map[col] = 'date'
            elif '主力净流入-净额' in col or ('主力' in col and '净额' in col):
                col_map[col] = 'main_net_inflow'
            elif '超大单净流入-净额' in col or ('超大单' in col and '净额' in col):
                col_map[col] = 'super_large_net'
            elif '大单净流入-净额' in col or ('大单' in col and '净额' in col):
                col_map[col] = 'large_net'
            elif '中单净流入-净额' in col or ('中单' in col and '净额' in col):
                col_map[col] = 'medium_net'
            elif '小单净流入-净额' in col or ('小单' in col and '净额' in col):
                col_map[col] = 'small_net'
            elif '主力净流入-净占比' in col or ('主力' in col and '占比' in col):
                col_map[col] = 'main_net_pct'
            elif '收盘价' in col or ('close' in col_lower and 'net' not in col_lower):
                col_map[col] = 'close'

        df = df.rename(columns=col_map)

        # 确保必要列存在
        needed = ['date', 'main_net_inflow']
        for col in needed:
            if col not in df.columns:
                print(f"[资金流] 缺少列: {col}, 可用列: {list(df.columns)}")
                return pd.DataFrame()

        # 转换数值
        for col in ['main_net_inflow', 'super_large_net', 'large_net',
                     'medium_net', 'small_net', 'main_net_pct', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        # 按日期过滤
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        df = df[mask].reset_index(drop=True)

        print(f"[资金流] 获取到 {len(df)} 天资金流数据")
        return df

    except Exception as e:
        print(f"[资金流] 获取失败: {e}")
        return pd.DataFrame()


def estimate_main_force_position(flow_df, df_price, initial_position_pct=0.3):
    """
    估算主力仓位和平均成本。

    方法：
      - 主力净流入为正 → 加仓，按当日收盘价计算加仓成本
      - 主力净流入为负 → 减仓，按先进先出(FIFO)或加权平均计算
      - 仓位 = 累计主力净流入 / 流通市值估算
      - 平均成本 = 累计买入金额 / 累计买入股数

    参数:
      flow_df: 资金流DataFrame
      df_price: 价格DataFrame
      initial_position_pct: 初始仓位假设（占流通市值比例）

    返回:
      DataFrame: 每日仓位估算
    """
    if flow_df is None or len(flow_df) == 0:
        return pd.DataFrame()

    df = flow_df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # 合并价格
    if 'close' not in df.columns or df['close'].isna().all():
        price_map = dict(zip(pd.to_datetime(df_price['date']), df_price['close']))
        df['close'] = df['date'].map(price_map)

    # 估算流通市值（用收盘价 * 一个估算系数）
    # 这里用主力净流入占比反推流通市值
    if 'main_net_pct' in df.columns and df['main_net_pct'].notna().any():
        # main_net_pct = 主力净流入 / 流通市值 * 100
        valid = df[df['main_net_pct'].abs() > 0.01]
        if len(valid) > 0:
            est_float_mv = (valid['main_net_inflow'].abs() / (valid['main_net_pct'].abs() / 100)).median()
        else:
            est_float_mv = df['close'].iloc[-1] * 1e8  # 粗略估算
    else:
        est_float_mv = df['close'].iloc[-1] * 1e8

    # 累计仓位和成本
    cum_inflow = 0.0      # 累计主力净流入（元）
    cum_buy_amount = 0.0   # 累计买入金额
    cum_buy_shares = 0.0   # 累计买入股数
    avg_cost = 0.0         # 平均成本
    position_pct = initial_position_pct  # 仓位占比

    positions = []
    for _, row in df.iterrows():
        close = row['close'] if pd.notna(row['close']) else 0
        net_inflow = row['main_net_inflow'] if pd.notna(row['main_net_inflow']) else 0

        if close > 0:
            if net_inflow > 0:
                # 加仓
                buy_shares = net_inflow / close
                cum_buy_amount += net_inflow
                cum_buy_shares += buy_shares
                avg_cost = cum_buy_amount / cum_buy_shares if cum_buy_shares > 0 else 0
            elif net_inflow < 0:
                # 减仓（按平均成本卖出）
                sell_shares = abs(net_inflow) / close
                cum_buy_shares = max(0, cum_buy_shares - sell_shares)
                if cum_buy_shares == 0:
                    cum_buy_amount = 0
                    avg_cost = 0
                else:
                    cum_buy_amount = cum_buy_shares * avg_cost

        cum_inflow += net_inflow
        # 仓位占比 = 初始仓位 + 累计净流入/流通市值
        position_pct = initial_position_pct + cum_inflow / est_float_mv
        position_pct = max(0, min(1, position_pct))

        # 浮动盈亏
        floating_pnl = (close - avg_cost) / avg_cost * 100 if avg_cost > 0 else 0

        positions.append({
            'date': row['date'],
            'close': round(close, 2),
            'main_net_inflow': round(net_inflow, 2),
            'main_net_pct': round(row.get('main_net_pct', 0), 4),
            'cum_inflow': round(cum_inflow, 2),
            'position_pct': round(position_pct * 100, 2),
            'avg_cost': round(avg_cost, 2),
            'floating_pnl_pct': round(floating_pnl, 2),
        })

    return pd.DataFrame(positions)


def identify_main_force_behavior(position_df, window=5):
    """
    识别主力买卖行为模式。

    模式分类：
      - 吸筹(accumulation): 价格横盘或微跌，主力持续净流入，仓位上升
      - 拉升(promotion): 价格快速上涨，主力净流入，仓位上升
      - 派发(distribution): 价格高位，主力净流出，仓位下降
      - 洗盘(shakeout): 价格下跌，主力净流入或小幅流出，仓位稳定
      - 观望(neutral): 无明显特征

    返回:
      DataFrame: 每日行为标签
    """
    if position_df is None or len(position_df) < window:
        return position_df

    df = position_df.copy()
    behaviors = []

    for i in range(len(df)):
        if i < window:
            behaviors.append('数据不足')
            continue

        recent = df.iloc[i - window:i + 1]
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0] * 100
        net_inflow_sum = recent['main_net_inflow'].sum()
        position_change = recent['position_pct'].iloc[-1] - recent['position_pct'].iloc[0]
        price_volatility = recent['close'].pct_change().std() * 100 if len(recent) > 1 else 0

        if price_change > 5 and net_inflow_sum > 0 and position_change > 0:
            behavior = '拉升'
        elif price_change < -5 and net_inflow_sum > 0 and abs(position_change) < 2:
            behavior = '洗盘'
        elif price_change < 3 and price_change > -3 and net_inflow_sum > 0 and position_change > 0:
            behavior = '吸筹'
        elif price_change > 0 and net_inflow_sum < 0 and position_change < 0:
            behavior = '派发'
        elif abs(price_change) < 2 and abs(net_inflow_sum) < 1e7:
            behavior = '观望'
        else:
            behavior = '混合'

        behaviors.append(behavior)

    df['behavior'] = behaviors
    return df


def analyze_capital_flow_patterns(flow_df, position_df, df_price, follow_days=3):
    """
    分析资金流-价格模式：资金流入/流出后价格延续概率。

    返回:
      dict: 模式分析结果
    """
    if flow_df is None or len(flow_df) == 0:
        return {'patterns': [], 'learning_updated': False}

    # 合并价格
    price_map = dict(zip(pd.to_datetime(df_price['date']), df_price['close']))
    flow_df = flow_df.copy()
    flow_df['close'] = flow_df['date'].map(price_map)

    patterns = []
    learning_updated = False

    for direction in ['inflow', 'outflow']:
        for strength_bin in ['weak', 'medium', 'strong']:
            # 筛选符合条件的日期
            if direction == 'inflow':
                mask = flow_df['main_net_inflow'] > 0
            else:
                mask = flow_df['main_net_inflow'] < 0

            if strength_bin == 'weak':
                strength_mask = flow_df['main_net_inflow'].abs() < 1e7
            elif strength_bin == 'medium':
                strength_mask = (flow_df['main_net_inflow'].abs() >= 1e7) & (flow_df['main_net_inflow'].abs() < 3e7)
            else:
                strength_mask = flow_df['main_net_inflow'].abs() >= 3e7

            selected = flow_df[mask & strength_mask]
            if len(selected) < 2:
                continue

            # 计算后续follow_days日价格变化
            followups = []
            for idx in selected.index:
                if idx + follow_days >= len(flow_df):
                    continue
                start_close = flow_df.loc[idx, 'close']
                end_close = flow_df.loc[idx + follow_days, 'close']
                if pd.notna(start_close) and pd.notna(end_close) and start_close > 0:
                    ret = (end_close - start_close) / start_close * 100
                    followups.append(ret)

            if len(followups) < 2:
                continue

            avg_followup = np.mean(followups)
            # 延续率：流入后上涨 / 流出后下跌
            if direction == 'inflow':
                continuation = sum(1 for r in followups if r > 0) / len(followups) * 100
            else:
                continuation = sum(1 for r in followups if r < 0) / len(followups) * 100

            strength_label = {'weak': '弱(<1千万)', 'medium': '中(1-3千万)', 'strong': '强(>3千万)'}[strength_bin]
            direction_label = '主力净流入' if direction == 'inflow' else '主力净流出'

            patterns.append({
                'direction': direction_label,
                'strength': strength_label,
                'count': len(followups),
                'avg_followup_pct': round(avg_followup, 4),
                'continuation_rate': round(continuation, 2),
                'std_followup': round(np.std(followups), 4),
            })

            # 持续学习
            try:
                for ret in followups:
                    flow_strength = selected['main_net_inflow'].abs().mean() / 1e7
                    learning.update_capital_flow_pattern(
                        code=flow_df['date'].iloc[0].strftime('%Y%m%d')[:6],
                        flow_direction=direction, flow_strength=flow_strength,
                        price_followup=ret, days=follow_days
                    )
                learning_updated = True
            except Exception:
                pass

    patterns.sort(key=lambda x: abs(x['avg_followup_pct']), reverse=True)
    return {'patterns': patterns, 'learning_updated': learning_updated, 'follow_days': follow_days}


def analyze_capital_flow_full(code, df_price, start_date, end_date):
    """
    完整的资金流分析流水线。
    """
    print(f"\n[资金流分析] 开始分析 {code}")

    # 1. 获取资金流数据
    flow_df = fetch_capital_flow_akshare(code, start_date, end_date)
    if len(flow_df) == 0:
        return {
            'flow_count': 0, 'flow_data': [], 'position_data': [],
            'behavior_data': [], 'patterns': [], 'learning_updated': False,
            'error': '未获取到资金流数据（可能数据源不可用）'
        }

    # 2. 估算主力仓位
    position_df = estimate_main_force_position(flow_df, df_price)

    # 3. 识别主力行为
    if len(position_df) > 5:
        position_df = identify_main_force_behavior(position_df)

    # 4. 资金流-价格模式分析
    pattern_result = analyze_capital_flow_patterns(flow_df, position_df, df_price)

    # 5. 持续学习：更新主力仓位
    try:
        if len(position_df) > 0:
            last = position_df.iloc[-1]
            learning.update_main_force_position(
                code=code, estimated_position=last['position_pct'],
                estimated_cost=last['avg_cost'], date=str(last['date'])
            )
    except Exception as e:
        print(f"[资金流] 仓位学习更新失败: {e}")

    # 6. 汇总
    flow_data = []
    for _, row in flow_df.tail(60).iterrows():
        flow_data.append({
            'date': str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date']),
            'close': round(row.get('close', 0), 2),
            'main_net_inflow': round(row.get('main_net_inflow', 0), 2),
            'main_net_pct': round(row.get('main_net_pct', 0), 4),
            'super_large_net': round(row.get('super_large_net', 0), 2),
            'large_net': round(row.get('large_net', 0), 2),
            'medium_net': round(row.get('medium_net', 0), 2),
            'small_net': round(row.get('small_net', 0), 2),
        })

    position_data = []
    for _, row in position_df.tail(60).iterrows():
        position_data.append({
            'date': str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date']),
            'close': row['close'],
            'position_pct': row['position_pct'],
            'avg_cost': row['avg_cost'],
            'floating_pnl_pct': row['floating_pnl_pct'],
            'behavior': row.get('behavior', ''),
            'main_net_inflow': row['main_net_inflow'],
        })

    # 行为统计
    behavior_stats = {}
    if 'behavior' in position_df.columns:
        for b, cnt in position_df['behavior'].value_counts().items():
            behavior_stats[b] = int(cnt)

    return {
        'flow_count': len(flow_df),
        'flow_data': flow_data,
        'position_data': position_data,
        'behavior_stats': behavior_stats,
        'patterns': pattern_result['patterns'],
        'learning_updated': pattern_result['learning_updated'],
        'current_position': position_data[-1] if position_data else None,
    }
