#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持续学习模块
============
存储和更新模型从历史数据中学到的参数：
  - 新闻敏感度：不同类型公告/新闻发布后的股价平均反应
  - 资金流模式：主力资金流入/流出后的价格延续概率
  - 指标权重更新：基于回测表现动态调整

学习数据以JSON文件存储在 learning/ 目录下，每次分析后自动更新。
"""
import os
import json
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEARNING_DIR = os.path.join(BASE_DIR, 'learning')
os.makedirs(LEARNING_DIR, exist_ok=True)


def _load(filename):
    """加载学习数据文件"""
    filepath = os.path.join(LEARNING_DIR, filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def _save(filename, data):
    """保存学习数据文件"""
    filepath = os.path.join(LEARNING_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return filepath


# ============================================================
# 新闻敏感度学习
# ============================================================
def update_news_sensitivity(code, news_type, timing, price_reaction, holding_days=5):
    """
    更新某只股票的新闻敏感度参数。

    参数:
      code: 股票代码
      news_type: 新闻类型（业绩预告/增持/减持/重组/监管/合同/其他）
      timing: 'intraday'(盘中) 或 'after_hours'(盘后)
      price_reaction: 新闻后N日的涨跌幅(%)
      holding_days: 持有天数
    """
    data = _load('news_sensitivity.json') or {}
    key = f"{code}_{news_type}_{timing}"
    if key not in data:
        data[key] = {
            'code': code, 'news_type': news_type, 'timing': timing,
            'count': 0, 'sum_reaction': 0.0, 'sum_sq': 0.0,
            'positive_count': 0, 'negative_count': 0,
            'avg_reaction': 0.0, 'std_reaction': 0.0,
            'win_rate': 0.0, 'last_updated': '',
            'history': []
        }
    entry = data[key]
    entry['count'] += 1
    entry['sum_reaction'] += price_reaction
    entry['sum_sq'] += price_reaction ** 2
    if price_reaction > 0:
        entry['positive_count'] += 1
    else:
        entry['negative_count'] += 1
    entry['avg_reaction'] = round(entry['sum_reaction'] / entry['count'], 4)
    if entry['count'] > 1:
        variance = entry['sum_sq'] / entry['count'] - entry['avg_reaction'] ** 2
        entry['std_reaction'] = round(max(np.sqrt(max(variance, 0)), 0.01), 4)
    entry['win_rate'] = round(entry['positive_count'] / entry['count'] * 100, 2)
    entry['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # 保留最近50条历史
    entry['history'].append({'date': datetime.now().strftime('%Y-%m-%d'),
                              'reaction': round(price_reaction, 4), 'holding_days': holding_days})
    entry['history'] = entry['history'][-50:]
    _save('news_sensitivity.json', data)
    return entry


def get_news_sensitivity(code, news_type=None, timing=None):
    """查询新闻敏感度学习结果"""
    data = _load('news_sensitivity.json') or {}
    results = []
    for key, entry in data.items():
        if entry['code'] != code:
            continue
        if news_type and entry['news_type'] != news_type:
            continue
        if timing and entry['timing'] != timing:
            continue
        results.append(entry)
    return results


def get_all_news_sensitivity():
    """获取全部新闻敏感度数据"""
    return _load('news_sensitivity.json') or {}


# ============================================================
# 资金流模式学习
# ============================================================
def update_capital_flow_pattern(code, flow_direction, flow_strength, price_followup, days=3):
    """
    更新资金流模式学习参数。

    参数:
      code: 股票代码
      flow_direction: 'inflow'(主力净流入) 或 'outflow'(主力净流出)
      flow_strength: 资金流强度（占流通市值比例%）
      price_followup: 后续N日涨跌幅(%)
      days: 跟踪天数
    """
    data = _load('capital_flow_patterns.json') or {}
    # 按强度分档：弱(<1%), 中(1-3%), 强(>3%)
    if abs(flow_strength) < 1:
        strength_bin = 'weak'
    elif abs(flow_strength) < 3:
        strength_bin = 'medium'
    else:
        strength_bin = 'strong'
    key = f"{code}_{flow_direction}_{strength_bin}"
    if key not in data:
        data[key] = {
            'code': code, 'flow_direction': flow_direction, 'strength_bin': strength_bin,
            'count': 0, 'sum_followup': 0.0, 'sum_sq': 0.0,
            'positive_count': 0, 'avg_followup': 0.0, 'std_followup': 0.0,
            'continuation_rate': 0.0, 'last_updated': '', 'history': []
        }
    entry = data[key]
    entry['count'] += 1
    entry['sum_followup'] += price_followup
    entry['sum_sq'] += price_followup ** 2
    # 延续率：资金流入后上涨 / 资金流出后下跌
    if (flow_direction == 'inflow' and price_followup > 0) or \
       (flow_direction == 'outflow' and price_followup < 0):
        entry['positive_count'] += 1
    entry['avg_followup'] = round(entry['sum_followup'] / entry['count'], 4)
    if entry['count'] > 1:
        variance = entry['sum_sq'] / entry['count'] - entry['avg_followup'] ** 2
        entry['std_followup'] = round(max(np.sqrt(max(variance, 0)), 0.01), 4)
    entry['continuation_rate'] = round(entry['positive_count'] / entry['count'] * 100, 2)
    entry['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry['history'].append({'date': datetime.now().strftime('%Y-%m-%d'),
                              'followup': round(price_followup, 4),
                              'flow_strength': round(flow_strength, 4)})
    entry['history'] = entry['history'][-50:]
    _save('capital_flow_patterns.json', data)
    return entry


def get_capital_flow_patterns(code):
    """查询某只股票的资金流模式学习结果"""
    data = _load('capital_flow_patterns.json') or {}
    return {k: v for k, v in data.items() if v['code'] == code}


# ============================================================
# 主力仓位学习（持续跟踪）
# ============================================================
def update_main_force_position(code, estimated_position, estimated_cost, date):
    """
    更新主力仓位估算（持续跟踪）。
    每次分析后更新，形成仓位变化轨迹。
    """
    data = _load('main_force_positions.json') or {}
    if code not in data:
        data[code] = {'code': code, 'positions': [], 'current_position': 0.0,
                       'current_cost': 0.0, 'last_updated': ''}
    entry = data[code]
    entry['positions'].append({'date': date, 'position': round(estimated_position, 4),
                                 'cost': round(estimated_cost, 4)})
    entry['positions'] = entry['positions'][-200:]  # 保留最近200条
    entry['current_position'] = round(estimated_position, 4)
    entry['current_cost'] = round(estimated_cost, 4)
    entry['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    _save('main_force_positions.json', data)
    return entry


def get_main_force_position(code):
    """查询主力仓位学习结果"""
    data = _load('main_force_positions.json') or {}
    return data.get(code, None)


# ============================================================
# 学习状态总览
# ============================================================
def get_learning_summary():
    """获取学习状态总览"""
    news = _load('news_sensitivity.json') or {}
    flows = _load('capital_flow_patterns.json') or {}
    positions = _load('main_force_positions.json') or {}
    return {
        'news_sensitivity_entries': len(news),
        'news_codes_learned': len(set(v['code'] for v in news.values())),
        'capital_flow_entries': len(flows),
        'flow_codes_learned': len(set(v['code'] for v in flows.values())),
        'position_codes_tracked': len(positions),
        'learning_dir': LEARNING_DIR
    }
