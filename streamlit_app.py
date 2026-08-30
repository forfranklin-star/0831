#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能量化分析系统 - Streamlit 版本
======================================
部署到 Streamlit Cloud 的主入口文件。

运行方式：
  streamlit run streamlit_app.py

功能模块：
  1. K线图与最优买卖点标注
  2. 最优交易明细
  3. 技术指标相关性分析
  4. 可解释数学模型
  5. 历史回测
  6. 预测准确率
  7. 实时买卖信号
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# 导入核心分析模块
import analysis

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="A股智能量化分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: 700;
        background: linear-gradient(90deg, #1976d2, #388e3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 13px;
        color: #666;
        margin-bottom: 20px;
    }
    .signal-buy {
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #ef5350;
    }
    .signal-sell {
        background: linear-gradient(135deg, #e0f2f1, #b2dfdb);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #26a69a;
    }
    .signal-hold {
        background: linear-gradient(135deg, #fff8e1, #ffecb3);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        border: 2px solid #ffa726;
    }
    .signal-text {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: 4px;
    }
    .metric-card {
        background: #f5f5f5;
        padding: 14px 18px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
    }
    .rule-item {
        background: #fafafa;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 6px;
        border-left: 3px solid #1976d2;
    }
    .formula-box {
        background: #263238;
        color: #a5d6a7;
        padding: 16px 20px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        line-height: 1.8;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 侧边栏 - 控制面板
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ 分析参数")

    stock_code = st.text_input(
        "股票代码",
        value="600519",
        help="输入6位A股代码，如 600519（贵州茅台）、000001（平安银行）、600118（中国卫星）"
    )

    # 默认日期范围：最近2年
    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("起始日期", value=pd.to_datetime(default_start))
    with col2:
        end_date = st.date_input("结束日期", value=pd.to_datetime(default_end))

    st.markdown("---")
    analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)
    signal_btn = st.button("📡 获取实时信号", use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚙️ 高级参数")
    min_profit = st.slider("最小交易收益率(%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                            help="过滤掉收益率低于此阈值的噪声交易")
    top_n_indicators = st.slider("模型Top指标数", min_value=3, max_value=10, value=6,
                                  help="选取相关性最高的N个指标构建模型")
    prob_threshold = st.slider("信号概率阈值", min_value=0.4, max_value=0.8, value=0.6, step=0.05,
                                help="买点/卖点概率超过此阈值才触发交易信号")

    st.markdown("---")
    st.caption("💡 数据来源：akshare（前复权）| 缓存24小时")


# ============================================================
# 主标题
# ============================================================
st.markdown('<div class="main-header">📈 A股智能量化分析系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">最优买卖点 · 技术指标相关性 · 可解释数学模型 · 历史回测 · 实时信号</div>', unsafe_allow_html=True)


# ============================================================
# 绘图工具函数
# ============================================================
def plot_kline(data, buy_markers, sell_markers):
    """绘制K线图，标注买卖点。"""
    dates = [d['date'] for d in data]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03, row_heights=[0.7, 0.3],
        subplot_titles=('K线与买卖点', '成交量')
    )

    # K线
    fig.add_trace(go.Candlestick(
        x=dates,
        open=[d['open'] for d in data],
        high=[d['high'] for d in data],
        low=[d['low'] for d in data],
        close=[d['close'] for d in data],
        name='K线',
        increasing_line_color='#ef5350',
        decreasing_line_color='#26a69a',
        increasing_fillcolor='#ef5350',
        decreasing_fillcolor='#26a69a'
    ), row=1, col=1)

    # 买入标记
    if buy_markers:
        buy_dates = [m['date'] for m in buy_markers]
        buy_prices = [m['price'] for m in buy_markers]
        fig.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices,
            mode='markers+text',
            marker=dict(symbol='triangle-up', size=14, color='#ef5350',
                        line=dict(color='#fff', width=1)),
            text=['▲'] * len(buy_markers),
            textposition='bottom center',
            textfont=dict(color='#ef5350', size=14),
            name=f'买入点({len(buy_markers)})',
            hovertemplate='买入: %{x}<br>价格: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    # 卖出标记
    if sell_markers:
        sell_dates = [m['date'] for m in sell_markers]
        sell_prices = [m['price'] for m in sell_markers]
        fig.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices,
            mode='markers+text',
            marker=dict(symbol='triangle-down', size=14, color='#26a69a',
                        line=dict(color='#fff', width=1)),
            text=['▼'] * len(sell_markers),
            textposition='top center',
            textfont=dict(color='#26a69a', size=14),
            name=f'卖出点({len(sell_markers)})',
            hovertemplate='卖出: %{x}<br>价格: %{y:.2f}<extra></extra>'
        ), row=1, col=1)

    # 成交量
    colors = ['#ef5350' if d['close'] >= d['open'] else '#26a69a' for d in data]
    fig.add_trace(go.Bar(
        x=dates, y=[d['volume'] for d in data],
        marker_color=colors, name='成交量', showlegend=False
    ), row=2, col=1)

    fig.update_layout(
        height=580,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_xaxes(type='category', tickangle=-45, row=2, col=1)
    fig.update_yaxes(title_text='价格', row=1, col=1)
    fig.update_yaxes(title_text='成交量', row=2, col=1)
    return fig


def plot_prob_curve(prob_curve):
    """绘制模型买卖概率曲线。"""
    dates = [p['date'] for p in prob_curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=[p['buy_prob'] for p in prob_curve],
        mode='lines', name='买点概率',
        line=dict(color='#ef5350', width=2),
        fill='tozeroy', fillcolor='rgba(239,83,80,0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=[p['sell_prob'] for p in prob_curve],
        mode='lines', name='卖点概率',
        line=dict(color='#26a69a', width=2),
        fill='tozeroy', fillcolor='rgba(38,166,154,0.1)'
    ))
    fig.add_hline(y=0.6, line_dash="dash", line_color="#ef5350",
                  annotation_text="买入阈值0.6", annotation_position="top right")
    fig.add_hline(y=0.6, line_dash="dash", line_color="#26a69a")
    fig.update_layout(
        height=350, hovermode='x unified',
        yaxis=dict(range=[0, 1], title='概率'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    return fig


def plot_correlation(corr_data, title):
    """绘制相关性条形图。"""
    top = corr_data[:15]
    indicators = [c['indicator'] for c in reversed(top)]
    corrs = [c['corr'] for c in reversed(top)]
    colors = ['#ef5350' if c['direction'] == 'high' else '#26a69a' for c in reversed(top)]

    fig = go.Figure(go.Bar(
        y=indicators, x=corrs, orientation='h',
        marker_color=colors,
        text=[f'{c:.3f}' for c in corrs],
        textposition='outside',
        hovertemplate='%{y}<br>相关系数: %{x:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=420,
        xaxis=dict(title='相关系数'),
        margin=dict(l=10, r=40, t=50, b=10),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    return fig


def plot_equity_curve(equity_curve, initial_capital):
    """绘制回测资金曲线。"""
    dates = [e['date'] for e in equity_curve]
    equity = [e['equity'] for e in equity_curve]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=equity, mode='lines', name='策略权益',
        line=dict(color='#1976d2', width=2),
        fill='tozeroy', fillcolor='rgba(25,118,210,0.08)'
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=[initial_capital] * len(dates),
        mode='lines', name='初始资金',
        line=dict(color='#999', width=1, dash='dash')
    ))
    fig.update_layout(
        height=420, hovermode='x unified',
        yaxis=dict(title='权益(元)', tickprefix='¥'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    return fig


def plot_profile(buy_profiles, buy_corr):
    """绘制买点前后Top5指标均值变化。"""
    top5 = [c['indicator'] for c in buy_corr[:5]]
    colors = ['#1976d2', '#388e3c', '#f57c00', '#7b1fa2', '#c62828']

    fig = go.Figure()
    for i, ind in enumerate(top5):
        if ind in buy_profiles:
            offsets = buy_profiles[ind]['offsets']
            values = buy_profiles[ind]['values']
            labels = [f't{o}' if o != 0 else 't(买点)' for o in offsets]
            fig.add_trace(go.Scatter(
                x=labels, y=values, mode='lines+markers',
                name=ind, line=dict(color=colors[i], width=2)
            ))
    fig.update_layout(
        title=dict(text='买点前后 Top5 指标均值变化 (t-3 ~ t+3)', font=dict(size=14)),
        height=380, hovermode='x unified',
        xaxis=dict(title='时间偏移'),
        yaxis=dict(title='指标均值'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    return fig


# ============================================================
# 结果展示函数
# ============================================================
def display_metrics_grid(metrics, optimal_return, buy_hold):
    """展示概览指标卡片。"""
    cols = st.columns(5)
    with cols[0]:
        st.metric("最优策略总收益", f"{optimal_return}%",
                  delta=f"vs买入持有 {optimal_return - buy_hold:.1f}%")
    with cols[1]:
        st.metric("模型回测收益", f"{metrics['total_return']}%",
                  delta=f"超额 {metrics['excess_return']}%")
    with cols[2]:
        st.metric("年化收益率", f"{metrics['annual_return']}%")
    with cols[3]:
        st.metric("最大回撤", f"{metrics['max_drawdown']}%")
    with cols[4]:
        st.metric("胜率", f"{metrics['win_rate']}%",
                  delta=f"{metrics['win_trades']}胜/{metrics['loss_trades']}负")


def display_rules(rules, title):
    """展示模型规则列表。"""
    st.markdown(f"**{title}**")
    for r in rules:
        direction_text = "偏高触发" if r['direction'] == 'high' else "偏低触发"
        direction_color = "#ef5350" if r['direction'] == 'high' else "#26a69a"
        st.markdown(f"""
        <div class="rule-item">
            <strong style="color:#1976d2;">{r['indicator']}</strong>
            <span style="background:{direction_color};color:#fff;padding:2px 8px;border-radius:4px;font-size:11px;margin-left:8px;">{direction_text}</span>
            <span style="float:right;background:#fff3e0;color:#e65100;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;">权重 {r['weight']}</span>
            <br><span style="font-size:12px;color:#666;">
                阈值: {r['threshold']:.4f} | 事件点均值: {r['mean_at_event']:.4f} | 非事件点均值: {r['mean_other']:.4f} | 标准差: {r['std']:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)


def display_signal(signal_data):
    """展示实时信号面板。"""
    signal = signal_data['signal']
    css_class = 'signal-buy' if signal == '买入' else ('signal-sell' if signal == '卖出' else 'signal-hold')
    signal_color = '#c62828' if signal == '买入' else ('#00695c' if signal == '卖出' else '#e65100')

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size:13px;color:#555;margin-bottom:8px;">当前操作建议</div>
            <div class="signal-text" style="color:{signal_color};">{signal}</div>
            <div style="margin-top:12px;font-size:14px;color:#333;">
                信号强度 <strong style="font-size:20px;color:{signal_color};">{signal_data['signal_strength']*100:.1f}%</strong>
            </div>
            <div style="display:flex;justify-content:center;gap:20px;margin-top:16px;">
                <div><div style="font-size:11px;color:#888;">买点概率</div><div style="font-size:18px;font-weight:700;color:#ef5350;">{signal_data['buy_probability']*100:.1f}%</div></div>
                <div><div style="font-size:11px;color:#888;">卖点概率</div><div style="font-size:18px;font-weight:700;color:#26a69a;">{signal_data['sell_probability']*100:.1f}%</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown(f"**📅 数据日期**: {signal_data['date']}")
        st.markdown(f"**💰 最新收盘**: ¥{signal_data['close']}")
        c1, c2, c3 = st.columns(3)
        c1.metric("开盘", f"¥{signal_data['open']}")
        c2.metric("最高", f"¥{signal_data['high']}")
        c3.metric("最低", f"¥{signal_data['low']}")

    with col2:
        st.markdown("**🔍 触发信号的关键指标状态**")
        for ind in signal_data['key_indicators']:
            status_text = "✓ 触发" if ind['triggered'] else "✗ 未触发"
            status_color = "#2e7d32" if ind['triggered'] else "#999"
            bg_color = "#e8f5e9" if ind['triggered'] else "#f5f5f5"
            direction_text = "偏高" if ind['direction'] == 'high' else "偏低"
            st.markdown(f"""
            <div style="background:{bg_color};padding:10px 14px;border-radius:6px;margin-bottom:6px;border-left:3px solid {'#4caf50' if ind['triggered'] else '#ccc'};">
                <strong>{ind['indicator']}</strong>
                <span style="float:right;color:{status_color};font-weight:600;font-size:13px;">{status_text}</span>
                <br><span style="font-size:12px;color:#555;">
                    当前值: <strong>{ind['value']:.4f}</strong> | 阈值: {ind['threshold']:.4f} | 方向: {direction_text} | 权重: {ind['weight']}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:11px;color:#999;margin-top:10px;">
            模型阈值: 买入≥{signal_data['model_threshold_buy']} | 卖出≥{signal_data['model_threshold_sell']}
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 主分析流程
# ============================================================
def run_analysis():
    """执行完整分析并展示结果。"""
    code = stock_code.strip()
    sd = start_date.strftime("%Y-%m-%d")
    ed = end_date.strftime("%Y-%m-%d")

    # 参数校验
    if not code or not code.isdigit() or len(code) != 6:
        st.error("❌ 请输入6位数字股票代码（如 600519）")
        return
    if start_date >= end_date:
        st.error("❌ 起始日期必须早于结束日期")
        return

    progress = st.progress(0, text="正在获取历史数据...")

    try:
        # 步骤1: 获取数据 + 计算指标
        progress.progress(15, text="正在获取历史日线数据（前复权）...")
        from analysis import fetch_stock_data, calc_indicators
        from datetime import timedelta as td
        fetch_start = (start_date - td(days=120)).strftime("%Y-%m-%d")
        df_raw = fetch_stock_data(code, fetch_start, ed)

        progress.progress(30, text="正在计算23个技术指标...")
        df_full = calc_indicators(df_raw)
        df = df_full[df_full['date'] >= sd].reset_index(drop=True)

        if len(df) < 10:
            st.error(f"❌ 选定时间段内有效数据不足（仅{len(df)}条），请扩大日期范围")
            return

        # 步骤2: 最优买卖点
        progress.progress(45, text="正在搜索最优买卖点...")
        from analysis import find_optimal_trades
        trades, total_return, buy_hold_return = find_optimal_trades(df, min_profit_pct=min_profit)

        # 步骤3: 相关性分析
        progress.progress(60, text="正在分析技术指标相关性...")
        from analysis import analyze_correlation
        buy_corr, sell_corr, buy_profiles, sell_profiles = analyze_correlation(df, trades)

        # 步骤4: 构建模型
        progress.progress(72, text="正在构建数学模型...")
        from analysis import build_model, compute_model_signals
        model = build_model(df, buy_corr, sell_corr, top_n=top_n_indicators)
        model['buy_threshold_prob'] = prob_threshold
        model['sell_threshold_prob'] = prob_threshold
        buy_probs, sell_probs = compute_model_signals(df, model)

        # 步骤5: 回测
        progress.progress(85, text="正在执行历史回测...")
        from analysis import backtest, evaluate_prediction_accuracy
        equity_curve, bt_trades, metrics = backtest(df, model, buy_probs, sell_probs)
        accuracy = evaluate_prediction_accuracy(df, buy_probs, sell_probs, horizon=5)

        progress.progress(95, text="正在生成可视化图表...")

        # 准备K线数据
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
        buy_markers = [{'date': t['buy_date'], 'price': t['buy_price']} for t in trades]
        sell_markers = [{'date': t['sell_date'], 'price': t['sell_price']} for t in trades]

        prob_curve = []
        for i in range(0, len(df), step):
            prob_curve.append({
                'date': str(df['date'].iloc[i]),
                'buy_prob': round(float(buy_probs[i]), 4),
                'sell_prob': round(float(sell_probs[i]), 4)
            })

        progress.progress(100, text="分析完成！")
        st.success(f"✅ 分析完成！股票 {code} | {sd} ~ {ed} | 共 {len(df)} 个交易日 | {len(trades)} 笔最优交易")

        # ===== 展示结果 =====
        st.markdown("---")
        display_metrics_grid(metrics, total_return, buy_hold_return)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 K线与买卖点", "📋 最优交易明细", "📊 指标相关性",
            "🧮 数学模型", "💰 回测结果", "🎯 预测准确率"
        ])

        # Tab1: K线
        with tab1:
            st.plotly_chart(plot_kline(kline_data, buy_markers, sell_markers), use_container_width=True)
            st.markdown("##### 模型买卖概率曲线")
            st.plotly_chart(plot_prob_curve(prob_curve), use_container_width=True)

        # Tab2: 最优交易
        with tab2:
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("最优交易笔数", len(trades))
            col_b.metric("复利总收益率", f"{total_return}%")
            col_c.metric("买入持有收益", f"{buy_hold_return}%")

            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.index = range(1, len(trades_df) + 1)
                trades_df.index.name = '序号'
                st.dataframe(trades_df, use_container_width=True, height=400)
            else:
                st.info("该时间段内未找到符合条件的交易")

        # Tab3: 相关性
        with tab3:
            col_l, col_r = st.columns(2)
            with col_l:
                st.plotly_chart(plot_correlation(buy_corr, "买点相关性排序（Top15）"), use_container_width=True)
            with col_r:
                st.plotly_chart(plot_correlation(sell_corr, "卖点相关性排序（Top15）"), use_container_width=True)

            st.plotly_chart(plot_profile(buy_profiles, buy_corr), use_container_width=True)

            # 相关性明细表
            with st.expander("📋 查看买点相关性完整数据"):
                st.dataframe(pd.DataFrame(buy_corr), use_container_width=True)
            with st.expander("📋 查看卖点相关性完整数据"):
                st.dataframe(pd.DataFrame(sell_corr), use_container_width=True)

        # Tab4: 数学模型
        with tab4:
            st.markdown(f"**模型类型**: {model['type']}")
            st.info(model['description'])

            st.markdown("##### 📐 模型公式")
            st.markdown(f'<div class="formula-box">{model["formula"]}</div>', unsafe_allow_html=True)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                display_rules(model['buy_rules'], "🟢 买点规则")
            with col_m2:
                display_rules(model['sell_rules'], "🔴 卖点规则")

        # Tab5: 回测
        with tab5:
            cols = st.columns(5)
            cols[0].metric("期末权益", f"¥{metrics['final_equity']:,.0f}")
            cols[1].metric("总收益率", f"{metrics['total_return']}%")
            cols[2].metric("年化收益率", f"{metrics['annual_return']}%")
            cols[3].metric("最大回撤", f"{metrics['max_drawdown']}%")
            cols[4].metric("盈亏比", metrics['profit_loss_ratio'])

            cols2 = st.columns(5)
            cols2[0].metric("交易次数", metrics['total_trades'])
            cols2[1].metric("盈利交易", metrics['win_trades'])
            cols2[2].metric("亏损交易", metrics['loss_trades'])
            cols2[3].metric("胜率", f"{metrics['win_rate']}%")
            cols2[4].metric("买入持有", f"{metrics['buy_hold_return']}%")

            st.plotly_chart(plot_equity_curve(equity_curve, analysis.INITIAL_CAPITAL), use_container_width=True)

            if bt_trades:
                bt_df = pd.DataFrame(bt_trades)
                st.dataframe(bt_df, use_container_width=True, height=350)
            else:
                st.info("回测期间未产生交易")

        # Tab6: 预测准确率
        with tab6:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.metric("买入预测准确率", f"{accuracy['buy_accuracy']}%",
                          delta=f"{accuracy['buy_signal_count']} 个买入信号")
            with col_a2:
                st.metric("卖出预测准确率", f"{accuracy['sell_accuracy']}%",
                          delta=f"{accuracy['sell_signal_count']} 个卖出信号")

            st.caption(f"预测窗口：未来 {accuracy['horizon']} 个交易日 | 上涨/下跌判定阈值：±1%")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("**买入信号样本**")
                if accuracy['buy_signals_sample']:
                    st.dataframe(pd.DataFrame(accuracy['buy_signals_sample']), use_container_width=True)
                else:
                    st.info("无买入信号样本")
            with col_s2:
                st.markdown("**卖出信号样本**")
                if accuracy['sell_signals_sample']:
                    st.dataframe(pd.DataFrame(accuracy['sell_signals_sample']), use_container_width=True)
                else:
                    st.info("无卖出信号样本")

    except ValueError as e:
        st.error(f"❌ {str(e)}")
    except Exception as e:
        st.error(f"❌ 分析过程出错: {str(e)}")
        st.exception(e)


# ============================================================
# 实时信号流程
# ============================================================
def run_signal():
    """获取并展示实时信号。"""
    code = stock_code.strip()
    if not code or not code.isdigit() or len(code) != 6:
        st.error("❌ 请输入6位数字股票代码")
        return

    with st.spinner("正在获取最新数据并计算信号..."):
        try:
            from analysis import generate_realtime_signal
            signal_data = generate_realtime_signal(code)
            st.success(f"✅ 实时信号获取成功 | 股票 {code} | 数据日期 {signal_data['date']}")
            display_signal(signal_data)
        except ValueError as e:
            st.error(f"❌ {str(e)}")
        except Exception as e:
            st.error(f"❌ 获取实时信号失败: {str(e)}")
            st.exception(e)


# ============================================================
# 按钮事件处理
# ============================================================
if analyze_btn:
    run_analysis()

if signal_btn:
    st.markdown("---")
    st.markdown("### 📡 实时买卖信号")
    run_signal()

# 默认提示
if not analyze_btn and not signal_btn:
    st.info("👈 在左侧输入股票代码和日期范围，点击「开始分析」或「获取实时信号」")
    st.markdown("---")
    st.markdown("""
    **功能说明：**
    - 📈 **K线与买卖点**：标注时间段内最优买卖点（红▲买/绿▼卖），展示模型概率曲线
    - 📋 **最优交易明细**：贪心峰谷法搜索的最大收益交易序列
    - 📊 **指标相关性**：23个技术指标与买卖点的点二列相关系数及显著性
    - 🧮 **数学模型**：阈值加权打分系统，明确指标、权重、规则
    - 💰 **回测结果**：100万初始资金，万2.5佣金+千1印花税，T+1执行
    - 🎯 **预测准确率**：买入/卖出信号后5日涨跌预测准确率
    """)
