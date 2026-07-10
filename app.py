import streamlit as st
import pandas as pd
import re
import numpy as np
from datetime import timedelta
import io

# 시각화 라이브러리
import plotly.graph_objects as go
import plotly.express as px

# --- 통계/시계열 라이브러리 ---
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# --- 머신러닝 라이브러리 (설치 여부 확인) ---
try:
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# --- 딥러닝 라이브러리 (설치 여부 확인) ---
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI 수요 예측 (스마트 필터)", layout="wide")


@st.cache_data
def load_data(uploaded_file):
    data = []
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')

    string_data = uploaded_file.getvalue().decode("utf-8")
    lines = string_data.split('\n')

    for line in lines:
        line = line.strip()
        if not line or "거래처명" in line or "거래내역조회" in line:
            continue
        match = date_pattern.search(line)
        if match:
            date_str = match.group(1)
            start_idx = match.start()
            end_idx = match.end()
            pre_date = line[:start_idx].strip()
            post_date = line[end_idx:].strip()

            parts_pre = pre_date.split(' ', 1)
            if len(parts_pre) >= 2 and parts_pre[0].isdigit():
                customer_name = parts_pre[1].strip()
            else:
                customer_name = pre_date.strip()

            if customer_name in ['(소계)', '(누계)', '소계', '누계']:
                continue

            parts_post = post_date.split()
            if parts_post:
                try:
                    qty = float(parts_post[0].replace(',', ''))
                    if qty > 0:
                        data.append({'Customer': customer_name, 'Date': date_str, 'Quantity': qty})
                except ValueError:
                    continue

    df = pd.DataFrame(data)
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
    return df


def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()


# -----------------------------------------------------------------------------
# 2. 거래처 등급 계산 함수 (NEW)
# -----------------------------------------------------------------------------
def calculate_customer_tiers(df):
    # 거래처별 총 거래량 집계
    cust_stats = df.groupby('Customer')['Quantity'].sum().reset_index()
    cust_stats = cust_stats.sort_values('Quantity', ascending=False)

    # 순위 백분율 계산
    cust_stats['Rank_Pct'] = cust_stats['Quantity'].rank(pct=True, ascending=False)

    # 등급 부여
    def get_tier(pct):
        if pct <= 0.1:
            return '💎 VIP (상위 10%)'
        elif pct <= 0.3:
            return '🥇 Gold (상위 30%)'
        elif pct <= 0.6:
            return '🥈 Silver (상위 60%)'
        else:
            return '🥉 Bronze (일반)'

    cust_stats['Tier'] = cust_stats['Rank_Pct'].apply(get_tier)
    return cust_stats[['Customer', 'Tier', 'Quantity']]


# -----------------------------------------------------------------------------
# 3. 예측 모델 함수 (추세 반영 강화)
# -----------------------------------------------------------------------------

def predict_linear_trend_force(series, weeks=5):
    try:
        lookback = 8
        recent_data = series[-lookback:] if len(series) >= lookback else series
        n = len(recent_data)
        if n < 2: return [series.mean()] * weeks

        x = np.arange(n)
        y = recent_data
        slope, intercept = np.polyfit(x, y, 1)

        future_x = np.arange(n, n + weeks)
        forecast = slope * future_x + intercept
        return forecast
    except:
        return [series.mean()] * weeks


def predict_holt_trend(series, weeks=5):
    try:
        if len(series) >= 4:
            model = ExponentialSmoothing(
                series, trend='add', seasonal=None, damped_trend=False
            ).fit(optimized=True)
            return model.forecast(weeks).values
        else:
            return predict_linear_trend_force(series, weeks)
    except:
        return predict_linear_trend_force(series, weeks)


def predict_arima_trend(series, weeks=5):
    try:
        model = ARIMA(series, order=(1, 1, 1)).fit()
        return model.forecast(steps=weeks).values
    except:
        return predict_linear_trend_force(series, weeks)


def create_ml_dataset(series, window_size=4):
    X, y = [], []
    s_list = list(series)
    for i in range(len(s_list) - window_size):
        X.append(s_list[i: i + window_size])
        y.append(s_list[i + window_size])
    return np.array(X), np.array(y)


def predict_ml_recursive(model, last_window, weeks=5):
    preds = []
    curr = list(last_window)
    for _ in range(weeks):
        in_row = np.array(curr[-len(last_window):]).reshape(1, -1)
        pred = model.predict(in_row)[0]
        preds.append(pred)
        curr.append(pred)
    return preds


def predict_rf(series, weeks=5):
    if not ML_AVAILABLE: return [0] * weeks
    win = 4
    if len(series) < win + 2: return predict_linear_trend_force(series, weeks)
    X, y = create_ml_dataset(series, win)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return predict_ml_recursive(model, series[-win:], weeks)


def predict_xgboost(series, weeks=5):
    if not ML_AVAILABLE: return [0] * weeks
    win = 4
    if len(series) < win + 2: return predict_linear_trend_force(series, weeks)
    X, y = create_ml_dataset(series, win)
    model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
    model.fit(X, y)
    return predict_ml_recursive(model, series[-win:], weeks)


def predict_lstm(series, weeks=5):
    if not TF_AVAILABLE: return [0] * weeks
    win = 4
    if len(series) < win + 5: return predict_linear_trend_force(series, weeks)
    scaler = MinMaxScaler(feature_range=(0, 1))
    s_scaled = scaler.fit_transform(np.array(series).reshape(-1, 1))
    X, y = [], []
    for i in range(len(s_scaled) - win):
        X.append(s_scaled[i: i + win])
        y.append(s_scaled[i + win])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(win, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=30, verbose=0, batch_size=2)

    preds = []
    curr = s_scaled[-win:]
    for _ in range(weeks):
        pred_sc = model.predict(curr.reshape(1, win, 1), verbose=0)[0][0]
        preds.append(pred_sc)
        curr = np.append(curr[1:], [[pred_sc]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


# -----------------------------------------------------------------------------
# 3. 메인 UI 구성
# -----------------------------------------------------------------------------
st.title("📈 AI 수요 예측 (스마트 필터)")
st.write("활동성(기간)과 거래 규모(VIP 등급) 필터를 활용하여 원하는 거래처를 쉽게 찾으세요.")

with st.sidebar:
    st.header("📂 1. 파일 업로드")
    uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df.empty:
        st.error("데이터 로드 실패")
    else:
        max_date = df['Date'].max()

        # [전처리] 거래처 등급 산출
        tier_df = calculate_customer_tiers(df)
        # 등급 정보를 메인 데이터프레임과 병합할 수도 있지만, 필터링용 리스트로 활용

        st.sidebar.success(f"로드 완료! ({len(df):,}건)")

        tab1, tab2 = st.tabs(["📊 상세 분석 (Ensemble)", "🏆 Top 거래처"])

        # TAB 1: 상세 분석
        with tab1:
            st.subheader("거래처 정밀 분석 및 필터링")

            # --- [필터 섹션] ---
            with st.expander("🔎 거래처 필터 옵션 (클릭하여 펼치기)", expanded=True):
                f_col1, f_col2 = st.columns(2)

                with f_col1:
                    st.markdown("##### 1. 활동성 기준 (Recency)")
                    activity_opt = st.radio(
                        "최근 거래일 기준:",
                        ('전체', '최근 3개월', '최근 6개월', '최근 1년'),
                        index=0,
                        horizontal=True
                    )

                with f_col2:
                    st.markdown("##### 2. 거래 규모 등급 (Volume)")
                    # 등급 목록 생성 (순서 보장)
                    tier_options = ['전체', '💎 VIP (상위 10%)', '🥇 Gold (상위 30%)', '🥈 Silver (상위 60%)', '🥉 Bronze (일반)']
                    tier_opt = st.selectbox("업체 등급 기준:", tier_options, index=0)

            # --- [필터링 로직 적용] ---
            # 1. 기간 필터링
            last_dates = df.groupby('Customer')['Date'].max().reset_index()
            if activity_opt == '최근 3개월':
                cutoff = max_date - timedelta(days=90)
                active_cust_list = last_dates[last_dates['Date'] >= cutoff]['Customer'].tolist()
            elif activity_opt == '최근 6개월':
                cutoff = max_date - timedelta(days=180)
                active_cust_list = last_dates[last_dates['Date'] >= cutoff]['Customer'].tolist()
            elif activity_opt == '최근 1년':
                cutoff = max_date - timedelta(days=365)
                active_cust_list = last_dates[last_dates['Date'] >= cutoff]['Customer'].tolist()
            else:
                active_cust_list = last_dates['Customer'].tolist()

            # 2. 등급 필터링
            if tier_opt != '전체':
                tier_cust_list = tier_df[tier_df['Tier'] == tier_opt]['Customer'].tolist()
            else:
                tier_cust_list = tier_df['Customer'].tolist()

            # 교집합 (두 조건 모두 만족)
            final_cust_list = list(set(active_cust_list) & set(tier_cust_list))
            final_cust_list.sort()

            # --- [선택 UI] ---
            st.divider()
            c_info, c_select = st.columns([1, 2])

            with c_info:
                st.metric("조건 만족 업체 수", f"{len(final_cust_list)} 개")
                if tier_opt != '전체':
                    st.caption(f"선택 등급: {tier_opt}")
                if activity_opt != '전체':
                    st.caption(f"활동 기간: {activity_opt} 이내")

            with c_select:
                search_txt = st.text_input("거래처명 검색", placeholder="예: 푸드")
                display_list = [c for c in final_cust_list if search_txt in c] if search_txt else final_cust_list
                selected_cust = st.selectbox("분석 대상 선택:", display_list, index=None, placeholder="목록에서 선택하세요...")

            # --- [분석 실행] ---
            if selected_cust:
                st.markdown(f"### 🎯 '{selected_cust}' 앙상블 예측")

                # 고객 등급 표시
                cust_tier_info = tier_df[tier_df['Customer'] == selected_cust]['Tier'].values[0]
                st.info(f"이 업체는 **{cust_tier_info}** 등급입니다.")

                cust_df = df[df['Customer'] == selected_cust].sort_values('Date')
                cust_weekly = cust_df.set_index('Date').resample('W-MON')['Quantity'].sum()

                start_date = cust_weekly[cust_weekly > 0].index.min()
                if pd.isna(start_date): start_date = cust_weekly.index.min()

                full_idx = pd.date_range(start=start_date, end=max_date, freq='W-MON')
                cust_weekly = cust_weekly.reindex(full_idx, fill_value=0)
                series_data = cust_weekly.values

                with st.spinner("예측 모델 분석 중..."):
                    p_linear = predict_linear_trend_force(series_data, 5)
                    p_holt = predict_holt_trend(series_data, 5)
                    p_arima = predict_arima_trend(series_data, 5)

                    p_rf = predict_rf(series_data, 5) if ML_AVAILABLE else [0] * 5
                    p_xgb = predict_xgboost(series_data, 5) if ML_AVAILABLE else [0] * 5
                    p_lstm = predict_lstm(series_data, 5) if TF_AVAILABLE else [0] * 5

                    valid_preds = [p_linear, p_holt, p_arima]
                    if ML_AVAILABLE: valid_preds.extend([p_rf, p_xgb])
                    if TF_AVAILABLE: valid_preds.append(p_lstm)

                    ens_pred = np.mean(valid_preds, axis=0)
                    ens_pred = [round(max(0, x), 1) for x in ens_pred]

                dates_str = [(max_date + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, 6)]

                res_df = pd.DataFrame({
                    '날짜': dates_str,
                    '앙상블(최종)': ens_pred,
                    '선형추세': [round(max(0, x), 1) for x in p_linear],
                    'Holt': [round(max(0, x), 1) for x in p_holt]
                })

                st.subheader("📋 예측 결과표")
                st.dataframe(res_df.set_index('날짜'), use_container_width=True)
                st.download_button("💾 엑셀 다운로드", data=to_excel(res_df), file_name=f"{selected_cust}_예측.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.markdown("---")
                st.subheader("📈 추세 그래프")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=cust_weekly.index[-20:], y=cust_weekly.values[-20:], name='과거 실적',
                                         line=dict(color='gray', dash='dot')))
                fig.add_trace(go.Scatter(x=pd.to_datetime(res_df['날짜']), y=res_df['앙상블(최종)'], name='앙상블(추세반영)',
                                         line=dict(color='red', width=4)))
                fig.add_trace(go.Scatter(x=pd.to_datetime(res_df['날짜']), y=res_df['선형추세'], name='선형추세',
                                         line=dict(color='blue', width=1, dash='dot')))
                fig.update_layout(height=500, hovermode="x unified", title=f"{selected_cust} 향후 5주 추세")
                st.plotly_chart(fig, use_container_width=True)

            else:
                # 미선택 시 목록 보여주기
                if display_list:
                    st.markdown(f"### 📋 조회된 거래처 목록 ({len(display_list)}개)")

                    # 상세 정보 병합 (마지막 거래일, 총거래량, 등급)
                    list_df = df[df['Customer'].isin(display_list)].groupby('Customer').agg(
                        마지막거래일=('Date', 'max'),
                        총거래량=('Quantity', 'sum')
                    ).reset_index()

                    # 등급 정보 병합
                    list_df = pd.merge(list_df, tier_df[['Customer', 'Tier']], on='Customer', how='left')
                    list_df['마지막거래일'] = list_df['마지막거래일'].dt.strftime('%Y-%m-%d')
                    list_df = list_df.sort_values(['총거래량', '마지막거래일'], ascending=[False, False])  # 거래량 순 정렬

                    st.dataframe(list_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("조건에 맞는 거래처가 없습니다. 필터를 변경해보세요.")

        # TAB 2: Top N
        with tab2:
            st.subheader("🏆 Top 거래처 일괄 예측")
            top_n = st.radio("분석 개수:", [10, 20, 30], horizontal=True)

            if st.button("Top N 분석 시작", type="primary"):
                top_customers = df.groupby('Customer')['Quantity'].sum().nlargest(top_n).index.tolist()
                results = []
                dates_str = [(max_date + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, 6)]

                bar = st.progress(0)
                for idx, cust in enumerate(top_customers):
                    c_df = df[df['Customer'] == cust].sort_values('Date')
                    c_weekly = c_df.set_index('Date').resample('W-MON')['Quantity'].sum()
                    start_d = c_weekly[c_weekly > 0].index.min()
                    if pd.isna(start_d): start_d = c_weekly.index.min()
                    c_series = c_weekly.reindex(pd.date_range(start=start_d, end=max_date, freq='W-MON'),
                                                fill_value=0).values

                    p1 = predict_linear_trend_force(c_series, 5)
                    p2 = predict_holt_trend(c_series, 5)
                    avg = np.mean([p1, p2], axis=0)

                    row = {'거래처명': cust}
                    for i, d in enumerate(dates_str): row[f'{i + 1}주차 ({d})'] = round(max(0, avg[i]), 1)
                    results.append(row)
                    bar.progress((idx + 1) / top_n)

                top_df = pd.DataFrame(results)
                st.dataframe(top_df, use_container_width=True)
                st.download_button("💾 Top N 결과 엑셀 다운로드", data=to_excel(top_df), file_name=f"Top{top_n}_예측.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.markdown("---")
                if st.toggle("종합 그래프 보기", value=True):
                    plot_df = top_df.melt(id_vars=['거래처명'], var_name='주차', value_name='예측수량')
                    fig2 = px.line(plot_df, x='주차', y='예측수량', color='거래처명', markers=True, title=f"Top {top_n} 거래처 추세")
                    fig2.update_layout(height=600, hovermode="x unified")
                    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 왼쪽 사이드바에서 파일을 업로드해주세요.")