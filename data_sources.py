#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一多数据源管理器
==================
为行情、新闻、资金流三类数据提供统一接口，每个数据源按优先级依次尝试，
第一个成功的就用。所有数据源失败时返回空结果而非抛出异常。

数据源优先级（海外服务器优化）：
  行情：Yahoo Finance → baostock → 腾讯财经 → efinance → AkShare-东财 → AkShare-新浪
  新闻：AkShare-东财 → efinance-同花顺 → AkShare-新浪 → 腾讯财经
  资金流：AkShare-东财 → efinance-同花顺 → AkShare-新浪
"""
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 数据源状态记录（用于调试和展示）
_source_status = {}


def _record_status(source_name, success, detail=''):
    """记录数据源调用状态"""
    _source_status[source_name] = {
        'success': success,
        'detail': detail,
        'time': datetime.now().strftime('%H:%M:%S')
    }


def get_source_status():
    """获取所有数据源调用状态"""
    return dict(_source_status)


def clear_source_status():
    """清空数据源状态记录"""
    _source_status.clear()


# ============================================================
# 新闻数据源
# ============================================================
def fetch_news(code, start_date, end_date, max_items=200, timeout=10):
    """
    多源获取个股新闻/公告，按优先级依次尝试。

    返回: DataFrame (date, title, url, source, news_type, timing)
    """
    clear_source_status()
    all_news = []

    # 源1: AkShare - 东方财富新闻
    try:
        import akshare as ak
        df = ak.stock_news_em(symbol=code)
        if df is not None and len(df) > 0:
            df = df.rename(columns={'发布时间': 'date', '新闻标题': 'title',
                                     '新闻链接': 'url', '文章来源': 'source'})
            count = 0
            for _, row in df.iterrows():
                try:
                    dt = pd.to_datetime(row['date'])
                    if pd.to_datetime(start_date) <= dt <= pd.to_datetime(end_date) + timedelta(days=1):
                        all_news.append({'date': dt, 'title': str(row['title']),
                                         'url': str(row.get('url', '')), 'source': '东方财富'})
                        count += 1
                except Exception:
                    continue
            _record_status('AkShare-东财新闻', True, f'{count}条')
            if count > 0:
                return _finalize_news(all_news, max_items)
        else:
            _record_status('AkShare-东财新闻', False, '返回空')
    except Exception as e:
        _record_status('AkShare-东财新闻', False, str(e)[:80])

    # 源2: efinance - 同花顺新闻
    try:
        import efinance as ef
        df = ef.stock.get_news_history(code)
        if df is not None and len(df) > 0:
            count = 0
            for _, row in df.iterrows():
                try:
                    dt = pd.to_datetime(row.get('发布时间', row.get('date', '')))
                    title = str(row.get('新闻标题', row.get('title', '')))
                    if pd.to_datetime(start_date) <= dt <= pd.to_datetime(end_date) + timedelta(days=1):
                        all_news.append({'date': dt, 'title': title,
                                         'url': '', 'source': '同花顺'})
                        count += 1
                except Exception:
                    continue
            _record_status('efinance-同花顺新闻', True, f'{count}条')
            if count > 0:
                return _finalize_news(all_news, max_items)
        else:
            _record_status('efinance-同花顺新闻', False, '返回空')
    except Exception as e:
        _record_status('efinance-同花顺新闻', False, str(e)[:80])

    # 源3: AkShare - 新浪财经新闻
    try:
        import akshare as ak
        df = ak.stock_news_sina(symbol=code)
        if df is not None and len(df) > 0:
            count = 0
            for _, row in df.iterrows():
                try:
                    dt = pd.to_datetime(row.get('发布时间', row.get('date', '')))
                    title = str(row.get('新闻标题', row.get('title', '')))
                    if pd.to_datetime(start_date) <= dt <= pd.to_datetime(end_date) + timedelta(days=1):
                        all_news.append({'date': dt, 'title': title,
                                         'url': str(row.get('新闻链接', '')), 'source': '新浪财经'})
                        count += 1
                except Exception:
                    continue
            _record_status('AkShare-新浪新闻', True, f'{count}条')
            if count > 0:
                return _finalize_news(all_news, max_items)
        else:
            _record_status('AkShare-新浪新闻', False, '返回空')
    except Exception as e:
        _record_status('AkShare-新浪新闻', False, str(e)[:80])

    # 源4: AkShare - 公告（巨潮资讯）
    try:
        import akshare as ak
        df = ak.stock_notice_report(symbol=code, date=end_date)
        if df is not None and len(df) > 0:
            count = 0
            for _, row in df.iterrows():
                try:
                    title = str(row.get('公告标题', row.get('标题', '')))
                    dt_str = str(row.get('公告日期', row.get('日期', '')))
                    dt = pd.to_datetime(dt_str)
                    if pd.to_datetime(start_date) <= dt <= pd.to_datetime(end_date) + timedelta(days=1):
                        all_news.append({'date': dt, 'title': title,
                                         'url': '', 'source': '巨潮公告'})
                        count += 1
                except Exception:
                    continue
            _record_status('AkShare-巨潮公告', True, f'{count}条')
            if count > 0:
                return _finalize_news(all_news, max_items)
        else:
            _record_status('AkShare-巨潮公告', False, '返回空')
    except Exception as e:
        _record_status('AkShare-巨潮公告', False, str(e)[:80])

    _record_status('全部新闻源', False, '4个源均失败')
    return pd.DataFrame(columns=['date', 'title', 'url', 'source', 'news_type', 'timing'])


def _finalize_news(all_news, max_items):
    """整理新闻结果：去重、分类、排序"""
    if not all_news:
        return pd.DataFrame(columns=['date', 'title', 'url', 'source', 'news_type', 'timing'])
    from news_analysis import classify_news_type, classify_news_timing
    df = pd.DataFrame(all_news)
    df = df.drop_duplicates(subset=['title']).sort_values('date').reset_index(drop=True)
    df = df.head(max_items)
    df['news_type'] = df['title'].apply(classify_news_type)
    df['timing'] = df['date'].apply(classify_news_timing)
    return df


# ============================================================
# 资金流数据源
# ============================================================
def fetch_capital_flow(code, start_date, end_date, timeout=10):
    """
    多源获取个股资金流数据，按优先级依次尝试。

    返回: DataFrame (date, main_net_inflow, super_large_net, large_net,
                     medium_net, small_net, main_net_pct, close)
    """
    # 判断市场
    if code.startswith(('60', '68', '90', '11', '13', '51', '58')):
        market = 'sh'
    else:
        market = 'sz'

    # 源1: AkShare - 东方财富资金流
    try:
        import akshare as ak
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        if df is not None and len(df) > 0:
            df = _standardize_flow_df(df)
            df = _filter_flow_dates(df, start_date, end_date)
            if len(df) > 0:
                _record_status('AkShare-东财资金流', True, f'{len(df)}天')
                return df
        _record_status('AkShare-东财资金流', False, '返回空或过滤后为空')
    except Exception as e:
        _record_status('AkShare-东财资金流', False, str(e)[:80])

    # 源2: efinance - 同花顺资金流
    try:
        import efinance as ef
        df = ef.stock.get_history_bill(code)
        if df is not None and len(df) > 0:
            # efinance列名映射
            col_map = {}
            for col in df.columns:
                if '日期' in str(col): col_map[col] = 'date'
                elif '主力净流入' in str(col): col_map[col] = 'main_net_inflow'
                elif '超大单净流入' in str(col): col_map[col] = 'super_large_net'
                elif '大单净流入' in str(col): col_map[col] = 'large_net'
                elif '中单净流入' in str(col): col_map[col] = 'medium_net'
                elif '小单净流入' in str(col): col_map[col] = 'small_net'
                elif '收盘' in str(col): col_map[col] = 'close'
            df = df.rename(columns=col_map)
            if 'date' in df.columns and 'main_net_inflow' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                for c in ['main_net_inflow', 'super_large_net', 'large_net', 'medium_net', 'small_net', 'close']:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors='coerce')
                if 'main_net_pct' not in df.columns:
                    df['main_net_pct'] = 0.0
                df = df.sort_values('date').reset_index(drop=True)
                df = _filter_flow_dates(df, start_date, end_date)
                if len(df) > 0:
                    _record_status('efinance-同花顺资金流', True, f'{len(df)}天')
                    return df
        _record_status('efinance-同花顺资金流', False, '返回空或列不匹配')
    except Exception as e:
        _record_status('efinance-同花顺资金流', False, str(e)[:80])

    # 源3: AkShare - 新浪资金流
    try:
        import akshare as ak
        df = ak.stock_individual_fund_flow_rank(indicator='今日')
        if df is not None and len(df) > 0:
            # 新浪资金流是全市场排名，需要筛选个股
            code_col = None
            for col in df.columns:
                if '代码' in str(col) or 'code' in str(col).lower():
                    code_col = col
                    break
            if code_col:
                df[code_col] = df[code_col].astype(str).str.zfill(6)
                df = df[df[code_col] == code]
                if len(df) > 0:
                    _record_status('AkShare-新浪资金流', True, f'{len(df)}条(仅当日)')
                    # 新浪只有当日数据，构造单日DataFrame
                    row = df.iloc[0]
                    today = datetime.now().strftime('%Y-%m-%d')
                    result = pd.DataFrame([{
                        'date': pd.to_datetime(today),
                        'main_net_inflow': pd.to_numeric(row.get('主力净流入-净额', 0), errors='coerce'),
                        'super_large_net': pd.to_numeric(row.get('超大单净流入-净额', 0), errors='coerce'),
                        'large_net': pd.to_numeric(row.get('大单净流入-净额', 0), errors='coerce'),
                        'medium_net': pd.to_numeric(row.get('中单净流入-净额', 0), errors='coerce'),
                        'small_net': pd.to_numeric(row.get('小单净流入-净额', 0), errors='coerce'),
                        'main_net_pct': pd.to_numeric(row.get('主力净流入-净占比', 0), errors='coerce'),
                        'close': pd.to_numeric(row.get('最新价', 0), errors='coerce'),
                    }])
                    return result
        _record_status('AkShare-新浪资金流', False, '未找到个股或返回空')
    except Exception as e:
        _record_status('AkShare-新浪资金流', False, str(e)[:80])

    _record_status('全部资金流源', False, '3个源均失败')
    return pd.DataFrame()


def _standardize_flow_df(df):
    """标准化资金流DataFrame列名"""
    col_map = {}
    for col in df.columns:
        col_str = str(col)
        if '日期' in col_str or col_str.lower() == 'date':
            col_map[col] = 'date'
        elif '主力净流入-净额' in col_str or ('主力' in col_str and '净额' in col_str):
            col_map[col] = 'main_net_inflow'
        elif '超大单净流入-净额' in col_str or ('超大单' in col_str and '净额' in col_str):
            col_map[col] = 'super_large_net'
        elif '大单净流入-净额' in col_str or ('大单' in col_str and '净额' in col_str):
            col_map[col] = 'large_net'
        elif '中单净流入-净额' in col_str or ('中单' in col_str and '净额' in col_str):
            col_map[col] = 'medium_net'
        elif '小单净流入-净额' in col_str or ('小单' in col_str and '净额' in col_str):
            col_map[col] = 'small_net'
        elif '主力净流入-净占比' in col_str or ('主力' in col_str and '占比' in col_str):
            col_map[col] = 'main_net_pct'
        elif '收盘价' in col_str or ('close' in col_str.lower() and 'net' not in col_str.lower()):
            col_map[col] = 'close'
    df = df.rename(columns=col_map)
    for c in ['main_net_inflow', 'super_large_net', 'large_net', 'medium_net', 'small_net', 'main_net_pct', 'close']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True) if 'date' in df.columns else df


def _filter_flow_dates(df, start_date, end_date):
    """按日期过滤资金流数据"""
    if 'date' not in df.columns or len(df) == 0:
        return df
    mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
    return df[mask].reset_index(drop=True)


# ============================================================
# 实时行情数据源（用于实时信号）
# ============================================================
def fetch_realtime_quote(code):
    """
    多源获取实时行情，按优先级依次尝试。
    返回: dict (price, open, high, low, volume, amount, change_pct, source)
    """
    # 源1: 腾讯财经直连
    try:
        import urllib.request
        market = 'sh' if code.startswith(('60', '68', '90', '11', '13', '51', '58')) else 'sz'
        url = f"http://qt.gtimg.cn/q={market}{code}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=8)
        text = resp.read().decode('gbk')
        parts = text.split('~')
        if len(parts) > 30:
            result = {
                'price': float(parts[3]),
                'open': float(parts[5]),
                'high': float(parts[33]),
                'low': float(parts[34]),
                'volume': float(parts[6]),
                'amount': float(parts[37]) if parts[37] else 0,
                'change_pct': float(parts[32]),
                'source': '腾讯财经'
            }
            _record_status('腾讯实时行情', True, f'¥{result["price"]}')
            return result
        _record_status('腾讯实时行情', False, '解析失败')
    except Exception as e:
        _record_status('腾讯实时行情', False, str(e)[:80])

    # 源2: AkShare - 东方财富实时行情
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is not None and len(df) > 0:
            code_col = None
            for col in df.columns:
                if '代码' in str(col):
                    code_col = col
                    break
            if code_col:
                df[code_col] = df[code_col].astype(str).str.zfill(6)
                row = df[df[code_col] == code]
                if len(row) > 0:
                    r = row.iloc[0]
                    result = {
                        'price': float(r.get('最新价', 0)),
                        'open': float(r.get('今开', 0)),
                        'high': float(r.get('最高', 0)),
                        'low': float(r.get('最低', 0)),
                        'volume': float(r.get('成交量', 0)),
                        'amount': float(r.get('成交额', 0)),
                        'change_pct': float(r.get('涨跌幅', 0)),
                        'source': '东方财富'
                    }
                    _record_status('AkShare-东财实时', True, f'¥{result["price"]}')
                    return result
        _record_status('AkShare-东财实时', False, '未找到个股')
    except Exception as e:
        _record_status('AkShare-东财实时', False, str(e)[:80])

    # 源3: Yahoo Finance
    try:
        import yfinance as yf
        ticker = f"{code}.SS" if code.startswith(('60', '68', '90')) else f"{code}.SZ"
        tk = yf.Ticker(ticker)
        info = tk.fast_info
        result = {
            'price': float(info.last_price),
            'open': float(info.open),
            'high': float(info.day_high),
            'low': float(info.day_low),
            'volume': float(info.last_volume),
            'amount': 0,
            'change_pct': 0,
            'source': 'Yahoo Finance'
        }
        _record_status('Yahoo实时行情', True, f'¥{result["price"]}')
        return result
    except Exception as e:
        _record_status('Yahoo实时行情', False, str(e)[:80])

    _record_status('全部实时行情源', False, '3个源均失败')
    return None
