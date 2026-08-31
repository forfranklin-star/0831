#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能量化分析系统 v2.0 - Streamlit 版本
============================================
增强功能：
  - 61个技术指标（交易员常用全覆盖）
  - 多周期：日线/60分钟/120分钟/30分钟/15分钟/周线
  - 大盘指数分析：上证/深证/创业板/沪深300/中证500等
  - 改进数学模型：全指标扫描→自动筛选→淘汰低相关→阈值加权
  - 模型灵敏度：保守/均衡/激进
  - 模型命名存储与加载
  - 实时信号（支持灵敏度和已保存模型）
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

import analysis

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="A股智能量化分析系统 v2.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size:26px; font-weight:700;
        background:linear-gradient(90deg,#1976d2,#388e3c); -webkit-background-clip:text;
        -webkit-text-fill-color:transparent; margin-bottom:2px; }
    .sub-header { font-size:12px; color:#666; margin-bottom:16px; }
    .signal-buy { background:linear-gradient(135deg,#ffebee,#ffcdd2); padding:20px;
        border-radius:12px; text-align:center; border:2px solid #ef5350; }
    .signal-sell { background:linear-gradient(135deg,#e0f2f1,#b2dfdb); padding:20px;
        border-radius:12px; text-align:center; border:2px solid #26a69a; }
    .signal-hold { background:linear-gradient(135deg,#fff8e1,#ffecb3); padding:20px;
        border-radius:12px; text-align:center; border:2px solid #ffa726; }
    .signal-text { font-size:36px; font-weight:800; letter-spacing:4px; }
    .formula-box { background:#263238; color:#a5d6a7; padding:14px 18px;
        border-radius:8px; font-family:'Courier New',monospace; font-size:12px;
        line-height:1.8; white-space:pre-wrap; }
    .rule-item { background:#fafafa; padding:10px 14px; border-radius:6px;
        margin-bottom:6px; border-left:3px solid #1976d2; }
    .eliminated-item { background:#f5f5f5; padding:6px 10px; border-radius:4px;
        margin-bottom:3px; font-size:12px; color:#999; }
</style>
""", unsafe_allow_html=True)

# 会话状态
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'signal_result' not in st.session_state:
    st.session_state.signal_result = None


# ============================================================
# 绘图函数
# ============================================================
def plot_kline(data, buy_markers, sell_markers):
    dates = [d['date'] for d in data]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.7, 0.3],
                        subplot_titles=('K线与买卖点', '成交量'))
    fig.add_trace(go.Candlestick(
        x=dates, open=[d['open'] for d in data], high=[d['high'] for d in data],
        low=[d['low'] for d in data], close=[d['close'] for d in data],
        name='K线', increasing_line_color='#ef5350', decreasing_line_color='#26a69a',
        increasing_fillcolor='#ef5350', decreasing_fillcolor='#26a69a'), row=1, col=1)
    if buy_markers:
        fig.add_trace(go.Scatter(
            x=[m['date'] for m in buy_markers], y=[m['price'] for m in buy_markers],
            mode='markers+text', marker=dict(symbol='triangle-up', size=14, color='#ef5350'),
            text=['▲']*len(buy_markers), textposition='bottom center',
            textfont=dict(color='#ef5350', size=14),
            name=f'买入点({len(buy_markers)})'), row=1, col=1)
    if sell_markers:
        fig.add_trace(go.Scatter(
            x=[m['date'] for m in sell_markers], y=[m['price'] for m in sell_markers],
            mode='markers+text', marker=dict(symbol='triangle-down', size=14, color='#26a69a'),
            text=['▼']*len(sell_markers), textposition='top center',
            textfont=dict(color='#26a69a', size=14),
            name=f'卖出点({len(sell_markers)})'), row=1, col=1)
    colors = ['#ef5350' if d['close'] >= d['open'] else '#26a69a' for d in data]
    fig.add_trace(go.Bar(x=dates, y=[d['volume'] for d in data],
                          marker_color=colors, name='成交量', showlegend=False), row=2, col=1)
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white', paper_bgcolor='white')
    fig.update_xaxes(type='category', tickangle=-45, row=2, col=1)
    return fig


def plot_prob_curve(prob_curve, threshold):
    dates = [p['date'] for p in prob_curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=[p['buy_prob'] for p in prob_curve],
                              mode='lines', name='买点概率', line=dict(color='#ef5350', width=2),
                              fill='tozeroy', fillcolor='rgba(239,83,80,0.1)'))
    fig.add_trace(go.Scatter(x=dates, y=[p['sell_prob'] for p in prob_curve],
                              mode='lines', name='卖点概率', line=dict(color='#26a69a', width=2),
                              fill='tozeroy', fillcolor='rgba(38,166,154,0.1)'))
    fig.add_hline(y=threshold, line_dash="dash", line_color="#666",
                  annotation_text=f"信号阈值{threshold}", annotation_position="top right")
    fig.update_layout(height=340, hovermode='x unified', yaxis=dict(range=[0, 1], title='概率'),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white', paper_bgcolor='white')
    return fig


def plot_correlation(corr_data, title, top_n=20):
    top = corr_data[:top_n]
    fig = go.Figure(go.Bar(
        y=[c['indicator'] for c in reversed(top)],
        x=[c['corr'] for c in reversed(top)], orientation='h',
        marker_color=['#ef5350' if c['direction']=='high' else '#26a69a' for c in reversed(top)],
        text=[f"{c['corr']:.3f}{'★' if c['significant'] else ''}" for c in reversed(top)],
        textposition='outside'))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=420,
                      xaxis=dict(title='相关系数'), margin=dict(l=10, r=50, t=50, b=10),
                      plot_bgcolor='white', paper_bgcolor='white')
    return fig


def plot_equity(equity_curve):
    dates = [e['date'] for e in equity_curve]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=[e['equity'] for e in equity_curve],
                              mode='lines', name='策略权益', line=dict(color='#1976d2', width=2),
                              fill='tozeroy', fillcolor='rgba(25,118,210,0.08)'))
    fig.add_trace(go.Scatter(x=dates, y=[analysis.INITIAL_CAPITAL]*len(dates),
                              mode='lines', name='初始资金', line=dict(color='#999', width=1, dash='dash')))
    fig.update_layout(height=400, hovermode='x unified',
                      yaxis=dict(title='权益(元)', tickprefix='¥'),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                      margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor='white', paper_bgcolor='white')
    return fig


# ============================================================
# 主标题
# ============================================================
st.markdown('<div class="main-header">📈 A股智能量化分析系统 v2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">61指标 · 多周期 · 大盘指数 · 自动筛选模型 · 灵敏度分级 · 模型存储 · 实时信号</div>', unsafe_allow_html=True)


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ 分析参数")

    analysis_type = st.radio("分析类型", ["个股", "大盘指数"], horizontal=True)

    if analysis_type == "大盘指数":
        index_options = {f"{v['name']}({k})": k for k, v in analysis.INDEX_MAP.items()}
        selected_idx = st.selectbox("选择指数", list(index_options.keys()))
        stock_code = index_options[selected_idx]
        is_index = True
    else:
        stock_code = st.text_input("股票代码", value="600519",
                                    help="6位代码，如 600519(茅台)、000001(平安银行)、600118(中国卫星)")
        is_index = False

    tf_options = {v['name']: k for k, v in analysis.TIMEFRAMES.items()}
    timeframe_name = st.selectbox("K线周期", list(tf_options.keys()), index=0)
    timeframe = tf_options[timeframe_name]

    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=730 if timeframe in ('daily','weekly') else 60)).strftime("%Y-%m-%d")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("起始日期", value=pd.to_datetime(default_start))
    with col_e:
        end_date = st.date_input("结束日期", value=pd.to_datetime(default_end))

    st.markdown("---")
    st.markdown("### 🧮 模型设置")

    sens_labels = {"conservative": "🛡️ 保守", "balanced": "⚖️ 均衡", "aggressive": "⚡ 激进"}
    sens_descs = {k: v['description'] for k, v in analysis.SENSITIVITY_CONFIG.items()}
    sensitivity = st.radio("模型灵敏度", list(sens_labels.keys()),
                            format_func=lambda x: sens_labels[x], horizontal=True)
    st.caption(sens_descs[sensitivity])

    min_profit = st.slider("最小交易收益率(%)", 0.5, 5.0, 1.0, 0.5)
    model_name = st.text_input("模型名称（可选）", value="",
                                help="输入名称后分析完成可保存该模型，用于后续实时信号调用")

    st.markdown("---")
    analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True)

    st.markdown("#### 📡 实时信号")
    saved_models_for_signal = analysis.list_saved_models()
    signal_model_options = ["自动构建模型（基于当前灵敏度）"]
    if st.session_state.current_model:
        signal_model_options.insert(0, f"📌 当前加载: {st.session_state.current_model['name']}")
    for m in saved_models_for_signal:
        signal_model_options.append(f"💾 {m['name']} ({m['sensitivity']})")
    selected_signal_model = st.selectbox("信号使用的模型", signal_model_options,
                                          help="选择已保存的模型生成信号，或自动构建")
    signal_btn = st.button("📡 获取实时信号", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 💾 已保存模型")
    saved_models = analysis.list_saved_models()
    if saved_models:
        for m in saved_models:
            with st.expander(f"{m['name']} ({m['sensitivity']})"):
                st.caption(f"创建: {m['created_at']}")
                st.caption(f"指标: 买{m['buy_indicators']}/卖{m['sell_indicators']}")
                if m.get('total_return') != '':
                    st.caption(f"回测: {m['total_return']}% | 胜率: {m['win_rate']}%")
                col_load, col_del = st.columns(2)
                if col_load.button("📂 加载", key=f"load_{m['name']}", use_container_width=True):
                    loaded_model, _, _ = analysis.load_model(m['name'])
                    st.session_state.current_model = loaded_model
                    st.success(f"已加载: {m['name']}")
                if col_del.button("🗑️ 删除", key=f"del_{m['name']}", use_container_width=True):
                    analysis.delete_model(m['name'])
                    st.rerun()
    else:
        st.caption("暂无保存的模型")

    if st.session_state.current_model:
        st.markdown(f"✅ 当前加载: **{st.session_state.current_model['name']}**")
        if st.button("清除加载模型", use_container_width=True):
            st.session_state.current_model = None
            st.rerun()


# ============================================================
# 分析执行
# ============================================================
def run_analysis():
    code = stock_code.strip()
    sd = start_date.strftime("%Y-%m-%d")
    ed = end_date.strftime("%Y-%m-%d")
    if not code or not code.isdigit() or len(code) != 6:
        st.error("❌ 请输入6位数字代码"); return
    if start_date >= end_date:
        st.error("❌ 起始日期必须早于结束日期"); return
    with st.spinner(f"正在分析 {code} ({timeframe_name})..."):
        try:
            result = analysis.run_full_analysis(
                code=code, start_date=sd, end_date=ed,
                timeframe=timeframe, is_index=is_index,
                sensitivity=sensitivity, model_name=model_name or None,
                min_profit_pct=min_profit)
            st.session_state.analysis_result = result
            st.session_state.current_model = result['model']
            st.success(f"✅ 分析完成！{code} | {timeframe_name} | {result['data_points']}根K线 | "
                       f"{result['trade_count']}笔最优交易 | 模型选中{len(result['model']['buy_rules'])}买/{len(result['model']['sell_rules'])}卖指标")
        except Exception as e:
            st.error(f"❌ 分析失败: {str(e)}"); st.exception(e)


def run_signal():
    code = stock_code.strip()
    if not code or not code.isdigit() or len(code) != 6:
        st.error("❌ 请输入6位数字代码"); return

    # 解析选择的模型
    signal_model = None
    sel = selected_signal_model
    if sel.startswith("📌 当前加载:"):
        signal_model = st.session_state.current_model
    elif sel.startswith("💾"):
        # 从 "💾 模型名 (灵敏度)" 中提取模型名
        model_name = sel[2:].split(' (')[0]
        loaded, _, _ = analysis.load_model(model_name)
        signal_model = loaded
        st.session_state.current_model = loaded
    # else: 自动构建，model=None

    with st.spinner(f"正在获取 {code} 实时信号（模型: {sel}）..."):
        try:
            sig = analysis.generate_realtime_signal(
                code=code, timeframe=timeframe, is_index=is_index,
                model=signal_model, sensitivity=sensitivity)
            st.session_state.signal_result = sig
            st.success(f"✅ {sig['signal']} | 强度{sig['signal_strength']*100:.1f}% | "
                       f"买{sig['buy_probability']*100:.1f}%/卖{sig['sell_probability']*100:.1f}% | "
                       f"模型: {sig.get('model_name','自动')}")
        except Exception as e:
            st.error(f"❌ 获取信号失败: {str(e)}"); st.exception(e)


if analyze_btn:
    run_analysis()
if signal_btn:
    run_signal()


# ============================================================
# 结果展示
# ============================================================
result = st.session_state.analysis_result

if result:
    model = result['model']
    metrics = result['metrics']

    cols = st.columns(6)
    cols[0].metric("最优策略收益", f"{result['optimal_total_return']}%",
                    delta=f"vs持有{result['optimal_buy_hold_return']}%")
    cols[1].metric("模型回测收益", f"{metrics['total_return']}%",
                    delta=f"超额{metrics['excess_return']}%")
    cols[2].metric("年化收益", f"{metrics['annual_return']}%")
    cols[3].metric("最大回撤", f"{metrics['max_drawdown']}%")
    cols[4].metric("胜率", f"{metrics['win_rate']}%",
                    delta=f"{metrics['win_trades']}胜{metrics['loss_trades']}负")
    cols[5].metric(f"模型({model['sensitivity_name']})",
                    f"买{len(model['buy_rules'])}/卖{len(model['sell_rules'])}指标",
                    delta=f"淘汰{len(model['buy_eliminated'])}个")

    if model_name:
        if st.button(f"💾 保存模型: {model_name}", type="primary"):
            analysis.save_model(model, metrics=metrics,
                                backtest_info={'code': stock_code, 'timeframe': timeframe,
                                               'start': result['start_date'], 'end': result['end_date']})
            st.success(f"✅ 模型 '{model_name}' 已保存！侧边栏可加载使用")

    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 K线与买卖点", "📋 最优交易", "📊 指标相关性",
        "🧮 数学模型", "💰 回测结果", "🎯 预测准确率"])

    with tab1:
        st.plotly_chart(plot_kline(result['kline_data'], result['buy_markers'], result['sell_markers']),
                        use_container_width=True)
        st.markdown("##### 模型买卖概率曲线")
        st.plotly_chart(plot_prob_curve(result['probability_curve'], model['buy_threshold_prob']),
                        use_container_width=True)

    with tab2:
        c1, c2, c3 = st.columns(3)
        c1.metric("最优交易笔数", result['trade_count'])
        c2.metric("复利总收益率", f"{result['optimal_total_return']}%")
        c3.metric("买入持有收益", f"{result['optimal_buy_hold_return']}%")
        if result['optimal_trades']:
            tdf = pd.DataFrame(result['optimal_trades'])
            tdf.index = range(1, len(tdf)+1); tdf.index.name = '序号'
            st.dataframe(tdf, use_container_width=True, height=400)
        else:
            st.info("无符合条件的交易")

    with tab3:
        cl, cr = st.columns(2)
        with cl:
            st.plotly_chart(plot_correlation(result['buy_correlation'],
                                              f"买点相关性Top20（共{len(result['buy_correlation'])}个指标）"),
                            use_container_width=True)
        with cr:
            st.plotly_chart(plot_correlation(result['sell_correlation'],
                                              f"卖点相关性Top20（共{len(result['sell_correlation'])}个指标）"),
                            use_container_width=True)
        with st.expander("📋 买点相关性完整数据"):
            st.dataframe(pd.DataFrame(result['buy_correlation']), use_container_width=True)
        with st.expander("📋 卖点相关性完整数据"):
            st.dataframe(pd.DataFrame(result['sell_correlation']), use_container_width=True)

    with tab4:
        st.markdown(f"**模型名称**: {model['name']}")
        st.markdown(f"**灵敏度**: {model['sensitivity_name']} ({sensitivity})")
        st.info(model['description'])
        cfg = model['sensitivity_config']
        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("最低|r|", cfg['min_corr'])
        cc2.metric("最高p值", cfg['max_pvalue'])
        cc3.metric("指标范围", f"{cfg['min_indicators']}-{cfg['max_indicators']}")
        cc4.metric("概率阈值", cfg['prob_threshold'])
        st.markdown("##### 📐 模型公式")
        st.markdown(f'<div class="formula-box">{model["formula"]}</div>', unsafe_allow_html=True)

        rl1, rl2 = st.columns(2)
        with rl1:
            st.markdown(f"##### 🟢 买点规则（选中{len(model['buy_rules'])}个）")
            for r in model['buy_rules']:
                dt = "偏高买入" if r['direction']=='high' else "偏低买入"
                dc = "#ef5350" if r['direction']=='high' else "#26a69a"
                st.markdown(f"""
                <div class="rule-item">
                    <strong style="color:#1976d2;">{r['indicator']}</strong>
                    <span style="background:{dc};color:#fff;padding:2px 6px;border-radius:3px;font-size:10px;margin-left:6px;">{dt}</span>
                    <span style="float:right;background:#fff3e0;color:#e65100;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:600;">权重{r['weight']}</span>
                    <br><span style="font-size:11px;color:#666;">
                        阈值:{r['threshold']:.4f} | r={r.get('corr',0):.3f} | p={r.get('pvalue',0):.4f}
                    </span>
                </div>""", unsafe_allow_html=True)
        with rl2:
            st.markdown(f"##### 🔴 卖点规则（选中{len(model['sell_rules'])}个）")
            for r in model['sell_rules']:
                dt = "偏高卖出" if r['direction']=='high' else "偏低卖出"
                dc = "#ef5350" if r['direction']=='high' else "#26a69a"
                st.markdown(f"""
                <div class="rule-item">
                    <strong style="color:#1976d2;">{r['indicator']}</strong>
                    <span style="background:{dc};color:#fff;padding:2px 6px;border-radius:3px;font-size:10px;margin-left:6px;">{dt}</span>
                    <span style="float:right;background:#fff3e0;color:#e65100;padding:2px 6px;border-radius:3px;font-size:11px;font-weight:600;">权重{r['weight']}</span>
                    <br><span style="font-size:11px;color:#666;">
                        阈值:{r['threshold']:.4f} | r={r.get('corr',0):.3f} | p={r.get('pvalue',0):.4f}
                    </span>
                </div>""", unsafe_allow_html=True)

        with st.expander(f"❌ 被淘汰指标（买点 {len(model['buy_eliminated'])} 个）"):
            for e in model['buy_eliminated'][:30]:
                st.markdown(f'<div class="eliminated-item">{e["indicator"]} | r={e["corr"]:.4f} | p={e["pvalue"]:.4f}</div>',
                            unsafe_allow_html=True)
        with st.expander(f"❌ 被淘汰指标（卖点 {len(model['sell_eliminated'])} 个）"):
            for e in model['sell_eliminated'][:30]:
                st.markdown(f'<div class="eliminated-item">{e["indicator"]} | r={e["corr"]:.4f} | p={e["pvalue"]:.4f}</div>',
                            unsafe_allow_html=True)

    with tab5:
        mc = st.columns(5)
        mc[0].metric("期末权益", f"¥{metrics['final_equity']:,.0f}")
        mc[1].metric("总收益率", f"{metrics['total_return']}%")
        mc[2].metric("年化收益率", f"{metrics['annual_return']}%")
        mc[3].metric("最大回撤", f"{metrics['max_drawdown']}%")
        mc[4].metric("盈亏比", metrics['profit_loss_ratio'])
        mc2 = st.columns(5)
        mc2[0].metric("交易次数", metrics['total_trades'])
        mc2[1].metric("盈利交易", metrics['win_trades'])
        mc2[2].metric("亏损交易", metrics['loss_trades'])
        mc2[3].metric("胜率", f"{metrics['win_rate']}%")
        mc2[4].metric("买入持有", f"{metrics['buy_hold_return']}%")
        st.plotly_chart(plot_equity(result['equity_curve']), use_container_width=True)
        if result['backtest_trades']:
            st.dataframe(pd.DataFrame(result['backtest_trades']), use_container_width=True, height=350)
        else:
            st.info("回测期间未产生交易")

    with tab6:
        acc = result['accuracy']
        ac1, ac2 = st.columns(2)
        ac1.metric("买入预测准确率", f"{acc['buy_accuracy']}%", delta=f"{acc['buy_signal_count']}个信号")
        ac2.metric("卖出预测准确率", f"{acc['sell_accuracy']}%", delta=f"{acc['sell_signal_count']}个信号")
        st.caption(f"预测窗口：未来{acc['horizon']}个周期 | 涨跌判定：±1%")
        as1, as2 = st.columns(2)
        with as1:
            st.markdown("**买入信号样本**")
            if acc['buy_signals_sample']:
                st.dataframe(pd.DataFrame(acc['buy_signals_sample']), use_container_width=True)
            else:
                st.info("无样本")
        with as2:
            st.markdown("**卖出信号样本**")
            if acc['sell_signals_sample']:
                st.dataframe(pd.DataFrame(acc['sell_signals_sample']), use_container_width=True)
            else:
                st.info("无样本")


# ============================================================
# 实时信号
# ============================================================
def _render_indicator_card(ind, side='buy'):
    """渲染单个指标触发状态卡片"""
    triggered = ind['triggered']
    stt = "✓ 触发" if triggered else "✗ 未触发"
    stc = "#2e7d32" if triggered else "#999"
    bg = "#e8f5e9" if triggered else "#fafafa"
    bd = "#4caf50" if triggered else "#e0e0e0"
    dt = "偏高" if ind['direction'] == 'high' else "偏低"
    side_color = "#ef5350" if side == 'buy' else "#26a69a"
    return f"""
    <div style="background:{bg};padding:8px 12px;border-radius:6px;margin-bottom:5px;border-left:3px solid {bd};">
        <strong style="color:{side_color};">{ind['indicator']}</strong>
        <span style="float:right;color:{stc};font-weight:600;font-size:12px;">{stt}</span>
        <br><span style="font-size:11px;color:#555;">
            当前值:<strong>{ind['value']:.4f}</strong> | 阈值:{ind['threshold']:.4f} |
            方向:{dt} | 权重:{ind['weight']} | r={ind.get('corr',0):.3f}
        </span>
    </div>"""


if st.session_state.signal_result:
    sig = st.session_state.signal_result
    st.markdown("---")
    st.markdown("### 📡 实时买卖信号")

    sc = 'signal-buy' if sig['signal'] == '买入' else ('signal-sell' if sig['signal'] == '卖出' else 'signal-hold')
    scolor = '#c62828' if sig['signal'] == '买入' else ('#00695c' if sig['signal'] == '卖出' else '#e65100')

    # 顶部：信号大卡片 + 操作建议说明
    top1, top2 = st.columns([1, 2])
    with top1:
        st.markdown(f"""
        <div class="{sc}">
            <div style="font-size:12px;color:#555;margin-bottom:6px;">当前操作建议</div>
            <div class="signal-text" style="color:{scolor};">{sig['signal']}</div>
            <div style="margin-top:10px;font-size:13px;">
                信号强度 <strong style="font-size:18px;color:{scolor};">{sig['signal_strength']*100:.1f}%</strong>
            </div>
            <div style="display:flex;justify-content:center;gap:16px;margin-top:12px;">
                <div><div style="font-size:10px;color:#888;">买点概率</div><div style="font-size:16px;font-weight:700;color:#ef5350;">{sig['buy_probability']*100:.1f}%</div></div>
                <div><div style="font-size:10px;color:#888;">卖点概率</div><div style="font-size:16px;font-weight:700;color:#26a69a;">{sig['sell_probability']*100:.1f}%</div></div>
            </div>
        </div>""", unsafe_allow_html=True)

    with top2:
        st.markdown(f"""
        <div style="background:#f5f5f5;padding:14px 18px;border-radius:8px;border-left:4px solid {scolor};">
            <div style="font-size:13px;font-weight:600;color:#333;margin-bottom:6px;">📋 操作建议说明</div>
            <div style="font-size:13px;color:#444;line-height:1.7;">{sig.get('action_description', '')}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("")
        info1, info2, info3, info4 = st.columns(4)
        info1.caption(f"**数据时间**: {sig['date']}")
        info2.caption(f"**周期**: {sig.get('timeframe', '日线')}")
        info3.caption(f"**模型**: {sig.get('model_name', '自动构建')}")
        info4.caption(f"**收盘价**: ¥{sig['close']} | 量: {sig['volume']:,}")

    st.markdown("")

    # 底部：全部买点指标 + 全部卖点指标
    buy_inds = sig.get('buy_indicators', [])
    sell_inds = sig.get('sell_indicators', [])
    buy_trig = sig.get('buy_triggered_count', sum(1 for i in buy_inds if i['triggered']))
    sell_trig = sig.get('sell_triggered_count', sum(1 for i in sell_inds if i['triggered']))
    buy_total = sig.get('buy_total_count', len(buy_inds))
    sell_total = sig.get('sell_total_count', len(sell_inds))

    col_buy, col_sell = st.columns(2)

    with col_buy:
        st.markdown(f"#### 🟢 买点指标触发状态（{buy_trig}/{buy_total} 触发）")
        if buy_inds:
            for ind in buy_inds:
                st.markdown(_render_indicator_card(ind, 'buy'), unsafe_allow_html=True)
        else:
            st.info("模型无买点指标规则")

    with col_sell:
        st.markdown(f"#### 🔴 卖点指标触发状态（{sell_trig}/{sell_total} 触发）")
        if sell_inds:
            for ind in sell_inds:
                st.markdown(_render_indicator_card(ind, 'sell'), unsafe_allow_html=True)
        else:
            st.info("模型无卖点指标规则")


# 空状态
if not result and not st.session_state.signal_result:
    st.info("👈 左侧设置参数（个股/指数、周期、日期、灵敏度），点击「开始分析」或「获取实时信号」")
    st.markdown("""
    **v2.0 增强功能：**
    - 📊 **61个技术指标**：MACD/RSI/KDJ/布林/均线/ATR/ADX/SAR/BIAS/乖离/一目均衡/唐奇安/动量/资金流/MFI/CCI/WR/ROC/MOM/CMO/ULTOSC/PSY/VWAP/WMA/STD/DMA/TRIX/STOCHRSI 等全覆盖
    - ⏱️ **多周期**：日线/60分钟/120分钟/30分钟/15分钟/周线
    - 📈 **大盘指数**：上证/深证/创业板/沪深300/中证500/科创50/上证50/中证1000/中小板指
    - 🧮 **改进模型**：全指标相关性扫描→按灵敏度自动筛选→淘汰低相关指标→阈值加权打分
    - 🎚️ **灵敏度分级**：保守(|r|≥0.15,p≤0.01,阈值0.7)/均衡(|r|≥0.08,p≤0.05,阈值0.6)/激进(|r|≥0.04,p≤0.1,阈值0.5)
    - 💾 **模型存储**：命名保存模型，侧边栏管理，可加载用于实时信号
    - 📡 **实时信号**：基于最新行情+指定灵敏度/已保存模型计算买卖触发概率
    """)
