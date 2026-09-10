#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一多数据源管理器 v2.0
========================
经过实际测试验证的数据源优先级：
  新闻：东方财富直连API(已验证) → akshare(备选)
  资金流：东方财富直连API(已验证) → akshare(备选)
  实时行情：腾讯财经直连(已验证) → 东方财富直连

核心改进：东方财富直连API使用纯urllib实现，不依赖akshare，
在海外服务器（Streamlit Cloud）上可正常访问。
"""
import time
import json
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

_source_status = {}


def _record_status(source_name, success, detail=''):
    _source_status[source_name] = {
        'success': success, 'detail': detail,
        'time': datetime.now().strftime('%H:%M:%S')
    }


def get_source_status():
    return dict(_source_status)


def clear_source_status():
    _source_status.clear()


def _http_get(url, timeout=10, headers=None):
    """统一HTTP GET请求"""
    default_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': '*/*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    if headers:
        default_headers.update(headers)
    req = urllib.request.Request(url, headers=default_headers)
    resp = urllib.request.urlopen(req, timeout=timeout)
    return resp.read()


def _get_secid(code):
    """东方财富secid: 沪市1.代码, 深市0.代码"""
    if code.startswith(('60', '68', '90', '11', '13', '51', '58')):
        return f"1.{code}"
    return f"0.{code}"


def _retry(func, retries=5, delay=2.0):
    """带重试的函数执行器（5次重试，间隔2秒，应对东方财富API频率限制）"""
    last_error = None
    for attempt in range(retries):
        try:
            result = func()
            if result is not None and (not isinstance(result, pd.DataFrame) or len(result) > 0):
                return result
            last_error = Exception("返回空结果")
        except Exception as e:
            last_error = e
        if attempt < retries - 1:
            # 指数退避：2s, 4s, 6s, 8s
            time.sleep(delay * (attempt + 1))
    raise last_error if last_error else Exception("未知错误")


# ============================================================
# 新闻数据源
# ============================================================
def _fetch_news_eastmoney_direct(code, start_date, end_date, max_items=100):
    """
    东方财富新闻搜索直连API（已验证可用，不依赖akshare）。
    """
    try:
        param = {
            "uid": "", "keyword": code, "type": ["cmsArticleWebOld"],
            "client": "web", "clientType": "web", "clientVersion": "curr",
            "param": {"cmsArticleWebOld": {
                "searchScope": "default", "sort": "default",
                "pageIndex": 1, "pageSize": max_items, "preTag": "", "postTag": ""
            }}
        }
        param_json = json.dumps(param, ensure_ascii=False)
        encoded_param = urllib.parse.quote(param_json)
        url = f"https://search-api-web.eastmoney.com/search/jsonp?cb=jQuery&param={encoded_param}"

        t0 = time.time()
        raw = _http_get(url, timeout=12, headers={'Referer': 'https://so.eastmoney.com/'})
        text = raw.decode('utf-8')
        elapsed = time.time() - t0

        start = text.find('(') + 1
        end = text.rfind(')')
        if start <= 0 or end <= start:
            raise ValueError(f"JSONP解析失败: {text[:200]}")
        data = json.loads(text[start:end])

        articles = data.get('result', {}).get('cmsArticleWebOld', [])
        if not isinstance(articles, list) or len(articles) == 0:
            _record_status('东财直连新闻', False, f'返回空, {elapsed:.1f}s')
            return pd.DataFrame()

        from news_analysis import classify_news_type, classify_news_timing
        results = []
        for art in articles:
            try:
                dt = pd.to_datetime(art.get('date', ''))
                if pd.to_datetime(start_date) <= dt <= pd.to_datetime(end_date) + timedelta(days=1):
                    title = art.get('title', '').replace('<em>', '').replace('</em>', '')
                    results.append({
                        'date': dt, 'title': title, 'url': art.get('url', ''),
                        'source': '东方财富',
                        'news_type': classify_news_type(title),
                        'timing': classify_news_timing(dt),
                    })
            except Exception:
                continue

        if results:
            df = pd.DataFrame(results).drop_duplicates(subset=['title']).sort_values('date').reset_index(drop=True)
            _record_status('东财直连新闻', True, f'{len(df)}条, {elapsed:.1f}s')
            return df.head(max_items)
        _record_status('东财直连新闻', False, f'日期过滤后为空, {elapsed:.1f}s')
        return pd.DataFrame()
    except Exception as e:
        _record_status('东财直连新闻', False, f'{type(e).__name__}: {str(e)[:100]}')
        return pd.DataFrame()


def _fetch_news_akshare(code, start_date, end_date, max_items=100):
    """akshare东方财富新闻（备选）"""
    try:
        import akshare as ak
        t0 = time.time()
        df = ak.stock_news_em(symbol=code)
        elapsed = time.time() - t0
        if df is None or len(df) == 0:
            _record_status('akshare新闻', False, f'返回空, {elapsed:.1f}s')
            return pd.DataFrame()
        from news_analysis import classify_news_type, classify_news_timing
        df = df.rename(columns={'发布时间': 'date', '新闻标题': 'title', '新闻链接': 'url', '文章来源': 'source'})
        results = []
        for _, row in df.iterrows():
            try:
                dt = pd.to_datetime(row['date'])
                if pd.to_datetime(start_date) <= dt <= pd.to_datetime(end_date) + timedelta(days=1):
                    title = str(row['title'])
                    results.append({
                        'date': dt, 'title': title, 'url': str(row.get('url', '')),
                        'source': '东方财富', 'news_type': classify_news_type(title),
                        'timing': classify_news_timing(dt),
                    })
            except Exception:
                continue
        if results:
            result_df = pd.DataFrame(results).drop_duplicates(subset=['title']).sort_values('date').reset_index(drop=True)
            _record_status('akshare新闻', True, f'{len(result_df)}条, {elapsed:.1f}s')
            return result_df.head(max_items)
        _record_status('akshare新闻', False, f'过滤后空, {elapsed:.1f}s')
        return pd.DataFrame()
    except Exception as e:
        _record_status('akshare新闻', False, f'{type(e).__name__}: {str(e)[:80]}')
        return pd.DataFrame()


def fetch_news(code, start_date, end_date, max_items=100):
    """多源获取个股新闻：东方财富直连API(首选，3次重试) → akshare(备选)"""
    clear_source_status()
    # 首选：东方财富直连API（带3次重试）
    try:
        df = _retry(lambda: _fetch_news_eastmoney_direct(code, start_date, end_date, max_items), retries=3, delay=1.0)
        if len(df) > 0:
            return df
    except Exception:
        pass
    # 备选：akshare
    df = _fetch_news_akshare(code, start_date, end_date, max_items)
    if len(df) > 0:
        return df
    _record_status('全部新闻源', False, '2个源均失败')
    return pd.DataFrame(columns=['date', 'title', 'url', 'source', 'news_type', 'timing'])


def get_source_status_list():
    """获取数据源状态列表（用于前端展示）"""
    status = get_source_status()
    result = []
    for name, st in status.items():
        result.append({
            'source': name,
            'success': st['success'],
            'detail': st['detail'],
            'time': st['time']
        })
    return result


# ============================================================
# 资金流数据源
# ============================================================
def _fetch_flow_eastmoney_direct(code, start_date, end_date):
    """
    东方财富个股资金流直连API（已验证可用，不依赖akshare）。
    """
    try:
        secid = _get_secid(code)
        url = (f"https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?"
               f"lmt=0&klt=101&fields1=f1,f2,f3,f7&"
               f"fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65&"
               f"ut=b2884a393a59ad64002292a3e90d46a5&secid={secid}")
        t0 = time.time()
        raw = _http_get(url, timeout=12, headers={'Referer': 'https://data.eastmoney.com/'})
        data = json.loads(raw.decode('utf-8'))
        elapsed = time.time() - t0

        klines = data.get('data', {}).get('klines', [])
        if not klines:
            _record_status('东财直连资金流', False, f'返回空, {elapsed:.1f}s')
            return pd.DataFrame()

        records = []
        for line in klines:
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                dt = pd.to_datetime(parts[0])
                if pd.to_datetime(start_date) <= dt <= pd.to_datetime(end_date):
                    records.append({
                        'date': dt,
                        'main_net_inflow': float(parts[1]),
                        'small_net': float(parts[2]),
                        'medium_net': float(parts[3]),
                        'large_net': float(parts[4]),
                        'super_large_net': float(parts[5]),
                        'main_net_pct': float(parts[6]) if len(parts) > 6 else 0.0,
                        'close': float(parts[11]) if len(parts) > 11 else 0.0,
                    })
            except Exception:
                continue

        if records:
            df = pd.DataFrame(records).sort_values('date').reset_index(drop=True)
            _record_status('东财直连资金流', True, f'{len(df)}天, {elapsed:.1f}s')
            return df
        _record_status('东财直连资金流', False, f'过滤后空, {elapsed:.1f}s')
        return pd.DataFrame()
    except Exception as e:
        _record_status('东财直连资金流', False, f'{type(e).__name__}: {str(e)[:100]}')
        return pd.DataFrame()


def _fetch_flow_akshare(code, start_date, end_date):
    """akshare资金流（备选）"""
    try:
        import akshare as ak
        market = 'sh' if code.startswith(('60', '68', '90')) else 'sz'
        t0 = time.time()
        df = ak.stock_individual_fund_flow(stock=code, market=market)
        elapsed = time.time() - t0
        if df is None or len(df) == 0:
            _record_status('akshare资金流', False, f'返回空, {elapsed:.1f}s')
            return pd.DataFrame()
        col_map = {}
        for col in df.columns:
            cs = str(col)
            if '日期' in cs: col_map[col] = 'date'
            elif '主力净流入-净额' in cs: col_map[col] = 'main_net_inflow'
            elif '超大单净流入-净额' in cs: col_map[col] = 'super_large_net'
            elif '大单净流入-净额' in cs: col_map[col] = 'large_net'
            elif '中单净流入-净额' in cs: col_map[col] = 'medium_net'
            elif '小单净流入-净额' in cs: col_map[col] = 'small_net'
            elif '主力净流入-净占比' in cs: col_map[col] = 'main_net_pct'
            elif '收盘价' in cs: col_map[col] = 'close'
        df = df.rename(columns=col_map)
        if 'date' not in df.columns:
            _record_status('akshare资金流', False, '缺少date列')
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        for c in ['main_net_inflow', 'super_large_net', 'large_net', 'medium_net', 'small_net', 'main_net_pct', 'close']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
        df = df[mask].sort_values('date').reset_index(drop=True)
        if len(df) > 0:
            _record_status('akshare资金流', True, f'{len(df)}天, {elapsed:.1f}s')
        else:
            _record_status('akshare资金流', False, f'过滤后空, {elapsed:.1f}s')
        return df
    except Exception as e:
        _record_status('akshare资金流', False, f'{type(e).__name__}: {str(e)[:80]}')
        return pd.DataFrame()


def fetch_capital_flow(code, start_date, end_date):
    """多源获取个股资金流：东方财富直连API(首选，3次重试) → akshare(备选)"""
    # 首选：东方财富直连API（带3次重试，解决临时RemoteDisconnected）
    try:
        df = _retry(lambda: _fetch_flow_eastmoney_direct(code, start_date, end_date), retries=3, delay=1.5)
        if len(df) > 0:
            return df
    except Exception:
        pass
    # 备选：akshare
    df = _fetch_flow_akshare(code, start_date, end_date)
    if len(df) > 0:
        return df
    _record_status('全部资金流源', False, '2个源均失败')
    return pd.DataFrame()


# ============================================================
# 实时行情数据源
# ============================================================
def fetch_realtime_quote(code):
    """多源获取实时行情：腾讯财经直连 → 东方财富直连"""
    # 源1: 腾讯财经直连
    try:
        market = 'sh' if code.startswith(('60', '68', '90', '11', '13', '51', '58')) else 'sz'
        url = f"http://qt.gtimg.cn/q={market}{code}"
        raw = _http_get(url, timeout=8)
        text = raw.decode('gbk')
        parts = text.split('~')
        if len(parts) > 37:
            result = {
                'price': float(parts[3]), 'open': float(parts[5]),
                'high': float(parts[33]), 'low': float(parts[34]),
                'volume': float(parts[6]), 'amount': float(parts[37]) if parts[37] else 0,
                'change_pct': float(parts[32]), 'source': '腾讯财经'
            }
            _record_status('腾讯实时行情', True, f'¥{result["price"]}')
            return result
        _record_status('腾讯实时行情', False, '解析失败')
    except Exception as e:
        _record_status('腾讯实时行情', False, f'{type(e).__name__}: {str(e)[:60]}')

    # 源2: 东方财富直连
    try:
        secid = _get_secid(code)
        url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f43,f44,f45,f46,f47,f48,f57,f58,f170&ut=b2884a393a59ad64002292a3e90d46a5"
        raw = _http_get(url, timeout=8, headers={'Referer': 'https://quote.eastmoney.com/'})
        data = json.loads(raw.decode('utf-8'))
        d = data.get('data', {})
        if d and d.get('f43'):
            result = {
                'price': float(d['f43']) / 100, 'open': float(d['f46']) / 100,
                'high': float(d['f44']) / 100, 'low': float(d['f45']) / 100,
                'volume': float(d.get('f47', 0)), 'amount': float(d.get('f48', 0)),
                'change_pct': float(d.get('f170', 0)) / 100, 'source': '东方财富'
            }
            _record_status('东财实时行情', True, f'¥{result["price"]}')
            return result
        _record_status('东财实时行情', False, '无数据')
    except Exception as e:
        _record_status('东财实时行情', False, f'{type(e).__name__}: {str(e)[:60]}')

    _record_status('全部实时行情源', False, '2个源均失败')
    return None
