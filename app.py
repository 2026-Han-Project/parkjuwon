import streamlit as st
import pandas as pd
import re
import numpy as np
from datetime import timedelta
from typing import TypedDict, List
import io
import sqlite3
from pathlib import Path

# 시각화 라이브러리
import plotly.graph_objects as go
import plotly.express as px

# --- 통계/시계열 라이브러리 ---
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests

# --- 머신러닝 라이브러리 (설치 여부 확인) ---
try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.linear_model import LinearRegression
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

# --- Prophet 라이브러리 (Layer 1 기저모델, 설치 여부 확인) ---
try:
    import logging as _logging
    from prophet import Prophet

    _logging.getLogger('cmdstanpy').setLevel(_logging.WARNING)
    _logging.getLogger('prophet').setLevel(_logging.WARNING)

    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# --- TFT(Temporal Fusion Transformer) 라이브러리 (Layer 1 고도화, 논문용 성능 비교, 설치 여부 확인) ---
try:
    import torch
    import lightning.pytorch as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    _logging.getLogger('pytorch_lightning').setLevel(_logging.WARNING)
    _logging.getLogger('lightning').setLevel(_logging.WARNING)

    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False

# --- LangGraph 에이전트 라이브러리 (설치 여부 확인) ---
try:
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# --- SHAP 라이브러리 (설치 여부 확인) ---
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI 수요 예측 (빵집 스마트 필터)", layout="wide")


@st.cache_data
def load_data(uploaded_file):
    data = []
    date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2})')

    string_data = uploaded_file.getvalue().decode("utf-8")
    lines = string_data.split('\n')

    for line in lines:
        line = line.strip()
        if not line or "상품명" in line or "품목명" in line or "판매내역조회" in line:
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
                item_name = parts_pre[1].strip()
            else:
                item_name = pre_date.strip()

            if item_name in ['(소계)', '(누계)', '소계', '누계']:
                continue

            parts_post = post_date.split()
            if parts_post:
                try:
                    qty = float(parts_post[0].replace(',', ''))
                    if qty > 0:
                        data.append({'Item': item_name, 'Date': date_str, 'Quantity': qty})
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
# 2. 품목 등급 계산 함수
# -----------------------------------------------------------------------------
def calculate_item_tiers(df):
    # 품목별 총 판매량 집계
    item_stats = df.groupby('Item')['Quantity'].sum().reset_index()
    item_stats = item_stats.sort_values('Quantity', ascending=False)

    # 순위 백분율 계산
    item_stats['Rank_Pct'] = item_stats['Quantity'].rank(pct=True, ascending=False)

    # 등급 부여
    def get_tier(pct):
        if pct <= 0.1:
            return '💎 시그니처 메뉴 (상위 10%)'
        elif pct <= 0.3:
            return '🥇 인기 메뉴 (상위 30%)'
        elif pct <= 0.6:
            return '🥈 스테디셀러 (상위 60%)'
        else:
            return '🥉 일반 품목'

    item_stats['Tier'] = item_stats['Rank_Pct'].apply(get_tier)
    return item_stats[['Item', 'Tier', 'Quantity']]


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
            # statsmodels 버전에 따라 forecast()가 ndarray 또는 Series를 반환하므로
            # np.asarray로 통일한다 (.values는 ndarray에 없어 AttributeError 발생 가능).
            return np.asarray(model.forecast(weeks))
        else:
            return predict_linear_trend_force(series, weeks)
    except Exception:
        return predict_linear_trend_force(series, weeks)


def predict_arima_trend(series, weeks=5):
    try:
        model = ARIMA(series, order=(1, 1, 1)).fit()
        return np.asarray(model.forecast(steps=weeks))
    except Exception:
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


def predict_prophet(dated_series, periods, freq='D'):
    """Layer 1 기저모델(Prophet). dated_series는 DatetimeIndex를 가진 pd.Series여야
    실제 달력(연간·주간 계절성)을 학습할 수 있다. 실패·미설치·데이터 부족 시
    predict_linear_trend_force로 폴백한다."""
    values = np.asarray(dated_series.values if hasattr(dated_series, 'values') else dated_series, dtype=float)
    if not PROPHET_AVAILABLE or len(values) < 10 or not hasattr(dated_series, 'index'):
        return predict_linear_trend_force(values, periods)
    try:
        train_df = pd.DataFrame({'ds': dated_series.index, 'y': values})
        span_periods = len(train_df)
        is_weekly = freq.startswith('W')
        model = Prophet(
            weekly_seasonality=(not is_weekly),
            yearly_seasonality=(span_periods >= (104 if is_weekly else 365)),
            daily_seasonality=False,
        )
        model.fit(train_df)
        future = model.make_future_dataframe(periods=periods, freq=freq, include_history=False)
        forecast = model.predict(future)
        return forecast['yhat'].to_numpy()[:periods]
    except Exception:
        return predict_linear_trend_force(values, periods)


def predict_tft(dated_series, periods, freq='D'):
    """Layer 1 고도화 모델(TFT, Temporal Fusion Transformer) — 논문용 Prophet 대비
    성능 비교 목적. 단일 품목 시계열에 대해 짧게 학습(few-epoch)한다. 데이터가
    부족(encoder+prediction 구간 확보 불가)하거나 미설치·학습 실패 시
    predict_prophet으로 폴백한다."""
    values = np.asarray(dated_series.values if hasattr(dated_series, 'values') else dated_series, dtype=float)
    n = len(values)
    encoder_len = min(30, max(10, n - periods - 5))
    if not TFT_AVAILABLE or n < encoder_len + periods + 5:
        return predict_prophet(dated_series, periods, freq=freq)
    try:
        torch.manual_seed(42)
        df = pd.DataFrame({
            'time_idx': np.arange(n),
            'group': 'series',
            'value': values,
        })
        training_cutoff = df['time_idx'].max() - periods

        training = TimeSeriesDataSet(
            df[df.time_idx <= training_cutoff],
            time_idx='time_idx',
            target='value',
            group_ids=['group'],
            min_encoder_length=max(1, encoder_len // 2),
            max_encoder_length=encoder_len,
            min_prediction_length=1,
            max_prediction_length=periods,
            time_varying_unknown_reals=['value'],
            target_normalizer=GroupNormalizer(groups=['group']),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

        train_dataloader = training.to_dataloader(train=True, batch_size=16, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=16, num_workers=0)

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=8,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(),
            optimizer='adam',
            log_interval=-1,
        )

        trainer = pl.Trainer(
            max_epochs=8,
            accelerator='cpu',
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        raw_predictions = tft.predict(val_dataloader, mode='prediction')
        forecast = np.asarray(raw_predictions[0]).flatten()[:periods]
        if len(forecast) < periods:
            return predict_prophet(dated_series, periods, freq=freq)
        return forecast
    except Exception:
        return predict_prophet(dated_series, periods, freq=freq)


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
# 4. LangGraph 에이전트 (조회 → 예측 → 해석 → 브리핑)
# -----------------------------------------------------------------------------
# ※ Layer 2(이벤트 게이팅)·Layer 3(품목간 수요연쇄)·SHAP 기여도 분해는 아직 로드맵 개발 중이라
#   이 에이전트의 '해석' 단계는 현재 계산된 예측치의 통계(최근 평균 대비 변화율)만 근거로 사용한다.
#   이벤트·품목연쇄 파이프라인이 붙으면 _interpret_node 안에서 해당 기여도를 추가로 반영하면 된다.

class ForecastBriefState(TypedDict):
    item: str
    tier: str
    history: List[float]
    forecast: List[float]
    dates: List[str]
    recent_avg: float
    next_avg: float
    pct_change: float
    trend: str
    peak_week: str
    briefing: str


def _retrieve_node(state: ForecastBriefState) -> dict:
    """조회: 최근 8주 판매 이력만 추려 다음 단계로 전달."""
    history = state["history"]
    recent_history = history[-8:] if len(history) >= 8 else history
    return {"history": recent_history}


def _predict_node(state: ForecastBriefState) -> dict:
    """예측: Streamlit에서 이미 계산된 앙상블 예측치를 그대로 통과시킨다."""
    return {}


def _interpret_node(state: ForecastBriefState) -> dict:
    """해석: 최근 평균 대비 예측 평균의 변화율로 트렌드를 규정 (SHAP 기여도 분해의 임시 근사)."""
    history = state["history"]
    forecast = state["forecast"]
    recent_avg = float(np.mean(history)) if history else 0.0
    next_avg = float(np.mean(forecast)) if forecast else 0.0
    pct_change = ((next_avg - recent_avg) / recent_avg * 100) if recent_avg > 0 else 0.0

    if pct_change >= 10:
        trend = "증가"
    elif pct_change <= -10:
        trend = "감소"
    else:
        trend = "보합"

    peak_idx = int(np.argmax(forecast)) if forecast else 0
    peak_week = state["dates"][peak_idx] if state.get("dates") else ""

    return {
        "recent_avg": round(recent_avg, 1),
        "next_avg": round(next_avg, 1),
        "pct_change": round(pct_change, 1),
        "trend": trend,
        "peak_week": peak_week,
    }


def _brief_node(state: ForecastBriefState) -> dict:
    """브리핑: 해석 결과를 자연어 문장으로 변환 (LLM 미연결 상태의 템플릿 기반 생성)."""
    trend_phrase = {
        "증가": f"최근 평균 대비 {abs(state['pct_change'])}% 증가가 예상됩니다.",
        "감소": f"최근 평균 대비 {abs(state['pct_change'])}% 감소가 예상됩니다.",
        "보합": "최근 평균과 비슷한 수준을 유지할 것으로 예상됩니다.",
    }[state["trend"]]

    action_phrase = {
        "증가": "결품을 막기 위해 생산량을 여유 있게 준비하는 것을 권장합니다.",
        "감소": "과다생산·폐기를 줄이기 위해 생산량을 보수적으로 조정하는 것을 권장합니다.",
        "보합": "평소 생산 계획을 유지해도 무방합니다.",
    }[state["trend"]]

    briefing = (
        f"'{state['item']}'({state['tier']})은 향후 5주 평균 약 {state['next_avg']}개 판매가 예상되며, "
        f"{trend_phrase} 특히 {state['peak_week']} 주간에 가장 높은 수요가 예상됩니다. {action_phrase}\n\n"
        "※ 본 브리핑은 시계열 기저 예측(Layer 1) 통계 기준이며, 이벤트 우선 게이팅(H1)·품목간 수요연쇄(H2′)·"
        "SHAP 기여도 분해는 로드맵 개발 중으로 아직 반영되지 않았습니다."
    )
    return {"briefing": briefing}


@st.cache_resource
def build_briefing_agent():
    graph = StateGraph(ForecastBriefState)
    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("predict", _predict_node)
    graph.add_node("interpret", _interpret_node)
    graph.add_node("brief", _brief_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "predict")
    graph.add_edge("predict", "interpret")
    graph.add_edge("interpret", "brief")
    graph.add_edge("brief", END)

    return graph.compile()


# -----------------------------------------------------------------------------
# 5. 이벤트 우선 게이팅 (Layer 2, H1) — 통합 이벤트 데이터셋 기반 Walk-forward 백테스트
# -----------------------------------------------------------------------------
# 데이터셋의 미래(예: 다음 5주)의 실제 공휴일·날씨는 알 수 없으므로, '미래 예측'이 아니라
# 최근 N일을 떼어내 실제값과 비교하는 백테스트로 H1(이벤트 우선 예측이 기저 예측보다 정확한가)을 검증한다.

@st.cache_data
def load_event_dataset(uploaded_file):
    return pd.read_csv(uploaded_file, parse_dates=['date'])


def build_event_dummies(daily):
    """일별 이벤트 플래그(요일·공휴일·방학·시즌·날씨)를 0/1 더미로 변환."""
    e = pd.DataFrame(index=daily.index)
    e['is_weekend'] = daily['is_weekend'].astype(float)
    e['is_holiday'] = daily['is_holiday'].astype(float)
    e['is_vacation'] = daily['is_vacation'].astype(float)
    e['is_rain'] = (daily['precip_type'] == '비').astype(float)
    e['is_snow'] = (daily['precip_type'] == '눈').astype(float)
    e['is_christmas'] = (daily['season_period'] == '크리스마스시즌').astype(float)
    e['is_suneung'] = (daily['season_period'] == '수능시즌').astype(float)
    e['is_valentine'] = (daily['season_period'] == '밸런타인시즌').astype(float)
    e['is_chuseok'] = (daily['season_period'] == '추석시즌').astype(float)
    return e


def fit_event_elasticity(qty, events):
    """이벤트별 수요 탄력도(β_k)를 추정한다.

    국소 기저치(14일 중심 이동중앙값) 대비 실제값의 상대 편차를 이벤트 더미에 회귀시켜,
    'ŷ_final = ŷ_base × (1 + Σ βₖ·eₖ)' 융합식의 βₖ를 구한다.
    """
    baseline = qty.rolling(14, center=True, min_periods=7).median()
    baseline = baseline.bfill().ffill()
    pct_dev = (qty - baseline) / baseline.replace(0, np.nan)
    pct_dev = pct_dev.fillna(0.0)

    reg = LinearRegression(fit_intercept=False)
    reg.fit(events.values, pct_dev.values)
    return dict(zip(events.columns, reg.coef_))


def apply_event_gating(base_forecast, beta, test_events, tau):
    """이벤트 강도 점수가 임계치 τ를 넘는 날에만 이벤트 조정치를 예측에 반영(게이팅)."""
    cols = list(test_events.columns)
    event_score = test_events.values @ np.array([beta[c] for c in cols])
    gate_on = np.abs(event_score) >= tau
    multiplier = np.where(gate_on, 1 + event_score, 1.0)
    gated_forecast = np.clip(np.asarray(base_forecast) * multiplier, 0, None)
    return gated_forecast, event_score, gate_on


def compute_wape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(actual - pred)) / denom * 100)


# -----------------------------------------------------------------------------
# 6. 품목간 수요연쇄 (Layer 3, H2′) — 선행 카테고리 → 후행 카테고리 리드-래그 백테스트
# -----------------------------------------------------------------------------

def find_optimal_lag(leading, lagging, min_lag=1, max_lag=14):
    """leading(선행 카테고리 일별 판매량) ↔ lagging(후행 카테고리) 교차상관으로 최적 시차 L*를 탐색."""
    best_lag, best_corr = min_lag, 0.0
    corr_by_lag = {}
    for lag in range(min_lag, max_lag + 1):
        shifted = leading.shift(lag)
        valid = shifted.notna() & lagging.notna()
        if valid.sum() < 10:
            continue
        corr = np.corrcoef(shifted[valid], lagging[valid])[0, 1]
        corr_by_lag[lag] = corr
        if abs(corr) > abs(best_corr):
            best_lag, best_corr = lag, corr
    return best_lag, best_corr, corr_by_lag


def granger_pvalue(lagging, leading, lag):
    """leading이 lagging을 Granger 인과하는지 검정, p-value 반환 (실패 시 None)."""
    try:
        data = pd.concat([lagging, leading], axis=1).dropna()
        data.columns = ['y', 'x']
        result = grangercausalitytests(data, maxlag=[lag], verbose=False)
        return result[lag][0]['ssr_ftest'][1]
    except Exception:
        return None


def fit_item_chain_gain(lagging_train, leading_lagged_train):
    """후행 품목의 '자체 기저치 대비 잔차'를 선행 품목의 시차 신호로 회귀해 γ를 추정.

    'ŷ_final = ŷ_base × (1+Σβₖeₖ) + γ·x_item(t−L*)' 융합식의 가산항(γ·x_item)에 대응한다.
    """
    baseline = lagging_train.rolling(14, center=True, min_periods=7).median()
    baseline = baseline.bfill().ffill()
    residual = (lagging_train - baseline).fillna(0.0)

    reg = LinearRegression()
    X = leading_lagged_train.values.reshape(-1, 1)
    reg.fit(X, residual.values)
    return reg


def predict_item_chain(base_forecast, reg, leading_lagged_test):
    X = leading_lagged_test.values.reshape(-1, 1)
    residual_hat = reg.predict(X)
    return np.clip(np.asarray(base_forecast) + residual_hat, 0, None)


# -----------------------------------------------------------------------------
# 7. 실시간 이상탐지 (Isolation Forest + 3σ + EWMA)
# -----------------------------------------------------------------------------

def rolling_3sigma_flags(series, window=28, min_periods=10):
    """3-시그마 관리도. 당일 값이 자기 기준선에 섞이는 셀프마스킹을 막기 위해
    전날까지의 값으로만 중심선·표준편차를 계산한다."""
    prior = series.shift(1)
    roll_mean = prior.rolling(window, min_periods=min_periods).mean()
    roll_std = prior.rolling(window, min_periods=min_periods).std()
    upper = roll_mean + 3 * roll_std
    lower = (roll_mean - 3 * roll_std).clip(lower=0)
    is_anomaly = (series > upper) | (series < lower)
    return pd.DataFrame({'value': series, 'center': roll_mean, 'ucl': upper, 'lcl': lower,
                          'is_anomaly': is_anomaly.fillna(False)})


def ewma_flags(series, lam=0.2, L=3, window=28, min_periods=10):
    """EWMA 관리도 (지수가중이동평균 기반, 완만한 추세성 이상 탐지에 강함)."""
    prior = series.shift(1)
    ewma = series.ewm(alpha=lam, adjust=False).mean()
    roll_mean = prior.rolling(window, min_periods=min_periods).mean()
    roll_std = prior.rolling(window, min_periods=min_periods).std()
    limit = L * roll_std * np.sqrt(lam / (2 - lam))
    ucl = roll_mean + limit
    lcl = (roll_mean - limit).clip(lower=0)
    is_anomaly = (ewma > ucl) | (ewma < lcl)
    return pd.DataFrame({'value': series, 'ewma': ewma, 'center': roll_mean, 'ucl': ucl, 'lcl': lcl,
                          'is_anomaly': is_anomaly.fillna(False)})


def isolation_forest_flags(series, contamination=0.05):
    """Isolation Forest 기반 다변량 이상탐지 (값·요일·최근추세 대비 편차를 특징으로 사용).
    contamination 비율만큼은 항상 '상대적으로' 가장 이상한 날로 플래그되는 특성이 있다."""
    if not ML_AVAILABLE or len(series) < 30:
        return pd.Series(False, index=series.index)
    weekday = series.index.dayofweek.values
    roll_mean = series.rolling(7, min_periods=1).mean().values
    features = np.column_stack([series.values, weekday, series.values - roll_mean])
    model = IsolationForest(contamination=contamination, random_state=42)
    pred = model.fit_predict(features)
    return pd.Series(pred == -1, index=series.index)


# -----------------------------------------------------------------------------
# 8. SHAP 기여도 분석 (XAI)
# -----------------------------------------------------------------------------
# build_event_dummies()로 만든 이벤트 피처에 기온·추세를 더해 RandomForest로 학습하고,
# SHAP TreeExplainer로 각 예측을 '기저치 + 이벤트별 기여도'로 분해한다.
# 여기서 나오는 (피처, 기여도) 쌍은 LangGraph 에이전트(_interpret_node)의 해석 단계에
# 그대로 연결할 수 있도록 설계했다.

SHAP_LABELS = {
    'is_weekend': '주말', 'is_holiday': '공휴일', 'is_vacation': '방학',
    'is_rain': '비', 'is_snow': '눈', 'is_christmas': '크리스마스시즌',
    'is_suneung': '수능시즌', 'is_valentine': '밸런타인시즌', 'is_chuseok': '추석시즌',
    'temperature': '기온', 'trend': '추세',
}


def build_shap_features(daily):
    """이벤트 더미 + 기온 + 추세(day index)로 이루어진 SHAP 학습용 피처 행렬."""
    features = build_event_dummies(daily)
    features['temperature'] = daily['temperature']
    features['trend'] = np.arange(len(daily))
    return features


def train_shap_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def explain_with_shap(model, X_test):
    """TreeExplainer로 SHAP 값과 기저값(base value)을 계산."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    base_value = float(np.ravel(explainer.expected_value)[0])
    return shap_values, base_value


def format_shap_briefing(item, date, base_value, contribs, pred, actual, top_n=5):
    """SHAP 기여도를 '기저 + 이벤트별 기여도' 형태의 자연어 문장으로 변환."""
    top = contribs.reindex(contribs.abs().sort_values(ascending=False).index).head(top_n)
    parts = []
    for name, v in top.items():
        if abs(v) < 0.05:
            continue
        label = SHAP_LABELS.get(name, name)
        parts.append(f"{label} {v:+.1f}개")
    detail = ", ".join(parts) if parts else "뚜렷한 이벤트 기여 없음"
    return (
        f"'{item}' {date} 예측치는 {pred:.1f}개(실제 {actual:.0f}개)로, "
        f"기저치 {base_value:.1f}개에 {detail}가 더해진 값입니다."
    )


# -----------------------------------------------------------------------------
# 9. Grafana 모니터링 연동 — 이상탐지 결과를 SQLite로 내보내기
# -----------------------------------------------------------------------------
# Grafana가 grafana-sqlite-datasource 플러그인으로 이 DB 파일을 직접 읽는다.
# (app.py -> monitoring.db -> Grafana, 별도 API 서버 없이 파일 기반으로 연동)

MONITORING_DB_PATH = Path(__file__).parent / "monitoring.db"


def export_monitoring_metrics(target, sigma_df, ewma_df, iso_flags, combined, scope_df=None,
                               db_path=MONITORING_DB_PATH):
    """이상탐지 탭에서 계산된 결과를 monitoring.db의 daily_metrics 테이블로 내보낸다."""
    scope_map = {}
    if scope_df is not None and not scope_df.empty:
        scope_map = {
            pd.Timestamp(row['날짜']): row['구분'] for _, row in scope_df.iterrows()
        }

    rows = []
    for d in combined.index:
        rows.append({
            'date': d.strftime('%Y-%m-%d'),
            'target': target,
            'value': float(sigma_df.loc[d, 'value']),
            'center': None if pd.isna(sigma_df.loc[d, 'center']) else float(sigma_df.loc[d, 'center']),
            'ucl': None if pd.isna(sigma_df.loc[d, 'ucl']) else float(sigma_df.loc[d, 'ucl']),
            'lcl': None if pd.isna(sigma_df.loc[d, 'lcl']) else float(sigma_df.loc[d, 'lcl']),
            'ewma': None if pd.isna(ewma_df.loc[d, 'ewma']) else float(ewma_df.loc[d, 'ewma']),
            'is_anomaly_3sigma': int(bool(combined.loc[d, '3시그마'])),
            'is_anomaly_ewma': int(bool(combined.loc[d, 'EWMA'])),
            'is_anomaly_iso': int(bool(combined.loc[d, 'IsolationForest'])),
            'detection_count': int(combined.loc[d, '탐지방법수']),
            'is_anomaly': int(bool(combined.loc[d, '이상여부'])),
            'scope': scope_map.get(pd.Timestamp(d), ''),
            'exported_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
    out_df = pd.DataFrame(rows)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT NOT NULL,
                target TEXT NOT NULL,
                value REAL,
                center REAL,
                ucl REAL,
                lcl REAL,
                ewma REAL,
                is_anomaly_3sigma INTEGER,
                is_anomaly_ewma INTEGER,
                is_anomaly_iso INTEGER,
                detection_count INTEGER,
                is_anomaly INTEGER,
                scope TEXT,
                exported_at TEXT,
                PRIMARY KEY (date, target)
            )
        """)
        conn.execute("DELETE FROM daily_metrics WHERE target = ?", (target,))
        out_df.to_sql('daily_metrics', conn, if_exists='append', index=False)
        conn.commit()
    finally:
        conn.close()

    return len(out_df)


# -----------------------------------------------------------------------------
# 10. What-if 채팅 (D팀) — 키워드 기반 시나리오 파싱 + H1 게이팅 모델로 실계산
# -----------------------------------------------------------------------------
# 실제 LLM API 키가 연결돼 있지 않아, 자연어 이해는 키워드 매칭으로 단순화했다.
# 대신 응답에 쓰이는 예측치·권장 생산량은 build_event_dummies/fit_event_elasticity로
# 학습한 실제 H1 모델에서 계산한 값이다 (지어낸 숫자가 아님).

WHATIF_KEYWORD_MAP = {
    'is_rain': ['비', '우천', '강수'],
    'is_snow': ['눈', '폭설'],
    'is_holiday': ['공휴일', '연휴', '휴일'],
    'is_vacation': ['방학'],
    'is_weekend': ['주말', '토요일', '일요일'],
    'is_christmas': ['크리스마스', '성탄'],
    'is_suneung': ['수능'],
    'is_valentine': ['밸런타인', '발렌타인'],
    'is_chuseok': ['추석'],
}


def parse_whatif_keywords(text):
    """채팅 문구에서 이벤트 키워드를 인식해 0/1 플래그로 변환 (단순 키워드 매칭, NLU 아님)."""
    flags = {k: False for k in WHATIF_KEYWORD_MAP}
    for key, kws in WHATIF_KEYWORD_MAP.items():
        if any(kw in text for kw in kws):
            flags[key] = True
    return flags


# -----------------------------------------------------------------------------
# 3. 메인 UI 구성
# -----------------------------------------------------------------------------
st.title("🥐 동네 빵집 AI 수요 예측")
st.write("판매 활성도(기간)와 판매 규모(베스트셀러 등급) 필터를 활용하여 원하는 품목을 쉽게 찾으세요.")

with st.sidebar:
    st.header("📂 1. 파일 업로드")
    uploaded_file = st.file_uploader("판매내역 CSV 파일을 선택하세요", type=['csv'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    if df.empty:
        st.error("데이터 로드 실패")
    else:
        max_date = df['Date'].max()

        # [전처리] 품목 등급 산출
        tier_df = calculate_item_tiers(df)
        # 등급 정보를 메인 데이터프레임과 병합할 수도 있지만, 필터링용 리스트로 활용

        st.sidebar.success(f"로드 완료! ({len(df):,}건)")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 품목별 상세 분석 (Ensemble)", "🏆 베스트셀러 TOP N",
            "🌦 이벤트 우선 예측 (H1 검증)", "🔗 품목간 수요연쇄 (H2′ 검증)",
            "🚨 이상탐지", "🧮 SHAP 기여도", "💬 What-if 채팅"
        ])

        # TAB 1: 상세 분석
        with tab1:
            st.subheader("품목 정밀 분석 및 필터링")

            # --- [필터 섹션] ---
            with st.expander("🔎 품목 필터 옵션 (클릭하여 펼치기)", expanded=True):
                f_col1, f_col2 = st.columns(2)

                with f_col1:
                    st.markdown("##### 1. 판매 활성도 기준 (Recency)")
                    activity_opt = st.radio(
                        "최근 판매일 기준:",
                        ('전체', '최근 3개월', '최근 6개월', '최근 1년'),
                        index=0,
                        horizontal=True
                    )

                with f_col2:
                    st.markdown("##### 2. 판매 규모 등급 (Volume)")
                    # 등급 목록 생성 (순서 보장)
                    tier_options = ['전체', '💎 시그니처 메뉴 (상위 10%)', '🥇 인기 메뉴 (상위 30%)', '🥈 스테디셀러 (상위 60%)', '🥉 일반 품목']
                    tier_opt = st.selectbox("품목 등급 기준:", tier_options, index=0)

            # --- [필터링 로직 적용] ---
            # 1. 기간 필터링
            last_dates = df.groupby('Item')['Date'].max().reset_index()
            if activity_opt == '최근 3개월':
                cutoff = max_date - timedelta(days=90)
                active_item_list = last_dates[last_dates['Date'] >= cutoff]['Item'].tolist()
            elif activity_opt == '최근 6개월':
                cutoff = max_date - timedelta(days=180)
                active_item_list = last_dates[last_dates['Date'] >= cutoff]['Item'].tolist()
            elif activity_opt == '최근 1년':
                cutoff = max_date - timedelta(days=365)
                active_item_list = last_dates[last_dates['Date'] >= cutoff]['Item'].tolist()
            else:
                active_item_list = last_dates['Item'].tolist()

            # 2. 등급 필터링
            if tier_opt != '전체':
                tier_item_list = tier_df[tier_df['Tier'] == tier_opt]['Item'].tolist()
            else:
                tier_item_list = tier_df['Item'].tolist()

            # 교집합 (두 조건 모두 만족)
            final_item_list = list(set(active_item_list) & set(tier_item_list))
            final_item_list.sort()

            # --- [선택 UI] ---
            st.divider()
            c_info, c_select = st.columns([1, 2])

            with c_info:
                st.metric("조건 만족 품목 수", f"{len(final_item_list)} 개")
                if tier_opt != '전체':
                    st.caption(f"선택 등급: {tier_opt}")
                if activity_opt != '전체':
                    st.caption(f"판매 기간: {activity_opt} 이내")

            with c_select:
                search_txt = st.text_input("품목명 검색", placeholder="예: 식빵")
                display_list = [c for c in final_item_list if search_txt in c] if search_txt else final_item_list
                selected_item = st.selectbox("분석 대상 선택:", display_list, index=None, placeholder="목록에서 선택하세요...")

            # --- [분석 실행] ---
            if selected_item:
                st.markdown(f"### 🎯 '{selected_item}' 앙상블 예측")

                # 품목 등급 표시
                item_tier_info = tier_df[tier_df['Item'] == selected_item]['Tier'].values[0]
                st.info(f"이 품목은 **{item_tier_info}** 입니다.")

                item_df = df[df['Item'] == selected_item].sort_values('Date')
                item_weekly = item_df.set_index('Date').resample('W-MON')['Quantity'].sum()

                start_date = item_weekly[item_weekly > 0].index.min()
                if pd.isna(start_date): start_date = item_weekly.index.min()

                full_idx = pd.date_range(start=start_date, end=max_date, freq='W-MON')
                item_weekly = item_weekly.reindex(full_idx, fill_value=0)
                series_data = item_weekly.values

                with st.spinner("예측 모델 분석 중..."):
                    p_linear = predict_linear_trend_force(series_data, 5)
                    p_holt = predict_holt_trend(series_data, 5)
                    p_arima = predict_arima_trend(series_data, 5)
                    p_prophet = predict_prophet(item_weekly, 5, freq='W-MON') if PROPHET_AVAILABLE else [0] * 5
                    p_tft = predict_tft(item_weekly, 5, freq='W-MON') if TFT_AVAILABLE else [0] * 5

                    p_rf = predict_rf(series_data, 5) if ML_AVAILABLE else [0] * 5
                    p_xgb = predict_xgboost(series_data, 5) if ML_AVAILABLE else [0] * 5
                    p_lstm = predict_lstm(series_data, 5) if TF_AVAILABLE else [0] * 5

                    valid_preds = [p_linear, p_holt, p_arima]
                    if PROPHET_AVAILABLE: valid_preds.append(p_prophet)
                    if TFT_AVAILABLE: valid_preds.append(p_tft)
                    if ML_AVAILABLE: valid_preds.extend([p_rf, p_xgb])
                    if TF_AVAILABLE: valid_preds.append(p_lstm)

                    ens_pred = np.mean(valid_preds, axis=0)
                    ens_pred = [round(max(0, x), 1) for x in ens_pred]

                dates_str = [(max_date + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, 6)]

                res_df = pd.DataFrame({
                    '날짜': dates_str,
                    '앙상블(최종)': ens_pred,
                    '선형추세': [round(max(0, x), 1) for x in p_linear],
                    'Holt': [round(max(0, x), 1) for x in p_holt],
                    'Prophet': [round(max(0, x), 1) for x in p_prophet],
                    'TFT': [round(max(0, x), 1) for x in p_tft],
                })

                st.subheader("📋 예측 결과표")
                st.dataframe(res_df.set_index('날짜'), use_container_width=True)
                st.download_button("💾 엑셀 다운로드", data=to_excel(res_df), file_name=f"{selected_item}_예측.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.markdown("---")
                st.subheader("📈 추세 그래프")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=item_weekly.index[-20:], y=item_weekly.values[-20:], name='과거 판매실적',
                                         line=dict(color='gray', dash='dot')))
                fig.add_trace(go.Scatter(x=pd.to_datetime(res_df['날짜']), y=res_df['앙상블(최종)'], name='앙상블(추세반영)',
                                         line=dict(color='red', width=4)))
                fig.add_trace(go.Scatter(x=pd.to_datetime(res_df['날짜']), y=res_df['선형추세'], name='선형추세',
                                         line=dict(color='blue', width=1, dash='dot')))
                fig.update_layout(height=500, hovermode="x unified", title=f"{selected_item} 향후 5주 판매 추세")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("🤖 에이전트 브리핑 (조회 → 예측 → 해석 → 브리핑)")
                if LANGGRAPH_AVAILABLE:
                    with st.spinner("에이전트가 브리핑을 생성하는 중..."):
                        agent = build_briefing_agent()
                        agent_state = agent.invoke({
                            "item": selected_item,
                            "tier": item_tier_info,
                            "history": [float(v) for v in item_weekly.values],
                            "forecast": [float(v) for v in ens_pred],
                            "dates": dates_str,
                        })
                    st.chat_message("assistant").write(agent_state["briefing"])
                else:
                    st.warning("langgraph가 설치되어 있지 않습니다. `pip install langgraph` 후 에이전트 브리핑을 사용할 수 있습니다.")

            else:
                # 미선택 시 목록 보여주기
                if display_list:
                    st.markdown(f"### 📋 조회된 품목 목록 ({len(display_list)}개)")

                    # 상세 정보 병합 (마지막 판매일, 총판매량, 등급)
                    list_df = df[df['Item'].isin(display_list)].groupby('Item').agg(
                        마지막판매일=('Date', 'max'),
                        총판매량=('Quantity', 'sum')
                    ).reset_index()

                    # 등급 정보 병합
                    list_df = pd.merge(list_df, tier_df[['Item', 'Tier']], on='Item', how='left')
                    list_df['마지막판매일'] = list_df['마지막판매일'].dt.strftime('%Y-%m-%d')
                    list_df = list_df.sort_values(['총판매량', '마지막판매일'], ascending=[False, False])  # 판매량 순 정렬

                    st.dataframe(list_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("조건에 맞는 품목이 없습니다. 필터를 변경해보세요.")

        # TAB 2: Top N
        with tab2:
            st.subheader("🏆 베스트셀러 품목 일괄 예측")
            top_n = st.radio("분석 개수:", [10, 20, 30], horizontal=True)

            if st.button("Top N 분석 시작", type="primary"):
                top_items = df.groupby('Item')['Quantity'].sum().nlargest(top_n).index.tolist()
                results = []
                dates_str = [(max_date + timedelta(weeks=i)).strftime('%Y-%m-%d') for i in range(1, 6)]

                bar = st.progress(0)
                for idx, item in enumerate(top_items):
                    i_df = df[df['Item'] == item].sort_values('Date')
                    i_weekly = i_df.set_index('Date').resample('W-MON')['Quantity'].sum()
                    start_d = i_weekly[i_weekly > 0].index.min()
                    if pd.isna(start_d): start_d = i_weekly.index.min()
                    i_series = i_weekly.reindex(pd.date_range(start=start_d, end=max_date, freq='W-MON'),
                                                fill_value=0).values

                    p1 = predict_linear_trend_force(i_series, 5)
                    p2 = predict_holt_trend(i_series, 5)
                    avg = np.mean([p1, p2], axis=0)

                    row = {'품목명': item}
                    for i, d in enumerate(dates_str): row[f'{i + 1}주차 ({d})'] = round(max(0, avg[i]), 1)
                    results.append(row)
                    bar.progress((idx + 1) / top_n)

                top_df = pd.DataFrame(results)
                st.dataframe(top_df, use_container_width=True)
                st.download_button("💾 Top N 결과 엑셀 다운로드", data=to_excel(top_df), file_name=f"Top{top_n}_예측.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                st.markdown("---")
                if st.toggle("종합 그래프 보기", value=True):
                    plot_df = top_df.melt(id_vars=['품목명'], var_name='주차', value_name='예측수량')
                    fig2 = px.line(plot_df, x='주차', y='예측수량', color='품목명', markers=True, title=f"Top {top_n} 품목 판매 추세")
                    fig2.update_layout(height=600, hovermode="x unified")
                    st.plotly_chart(fig2, use_container_width=True)

        # TAB 3: 이벤트 우선 게이팅(H1) 검증
        with tab3:
            st.subheader("🌦 이벤트 우선 게이팅 (Layer 2, H1) — Walk-forward 백테스트")
            st.caption(
                "공휴일·방학·시즌·날씨 이벤트를 반영한 예측이, 이벤트를 무시한 기저 시계열 예측보다 "
                "실제로 더 정확한지 최근 기간을 떼어내 검증합니다 (가설 H1). "
                "미래 이벤트는 알 수 없으므로 '미래 예측'이 아니라 과거 데이터로 백테스트합니다."
            )
            event_file = st.file_uploader(
                "이벤트 통합 데이터셋 CSV 업로드 (date, item, sales_qty, is_weekend, is_holiday, "
                "is_vacation, season_period, precip_type 컬럼 포함)",
                type=['csv'], key='event_csv'
            )

            if event_file is not None:
                edf = load_event_dataset(event_file)
                required_cols = {'date', 'item', 'sales_qty', 'is_weekend', 'is_holiday',
                                  'is_vacation', 'season_period', 'precip_type'}
                missing_cols = required_cols - set(edf.columns)

                if missing_cols:
                    st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")
                elif not ML_AVAILABLE:
                    st.warning("scikit-learn이 설치되어 있지 않아 이벤트 탄력도를 추정할 수 없습니다.")
                else:
                    ev_items = sorted(edf['item'].unique())
                    ev_item = st.selectbox("품목 선택", ev_items, key='event_item')
                    horizon = st.slider("백테스트 검증 기간(일)", 7, 28, 14, key='event_horizon')
                    tau = st.slider(
                        "이벤트 게이팅 임계치 τ (이 값을 넘는 이벤트 강도만 예측에 반영)",
                        0.0, 0.3, 0.10, 0.01, key='event_tau'
                    )

                    item_daily = edf[edf['item'] == ev_item].groupby('date').agg(
                        sales_qty=('sales_qty', 'sum'),
                        is_weekend=('is_weekend', 'first'),
                        is_holiday=('is_holiday', 'first'),
                        is_vacation=('is_vacation', 'first'),
                        season_period=('season_period', 'first'),
                        precip_type=('precip_type', 'first'),
                    ).sort_index()

                    if len(item_daily) < horizon + 30:
                        st.warning("검증에 필요한 데이터(최소 학습 30일 + 검증기간)가 부족합니다.")
                    else:
                        train = item_daily.iloc[:-horizon]
                        test = item_daily.iloc[-horizon:]

                        events = build_event_dummies(item_daily)
                        beta = fit_event_elasticity(train['sales_qty'], events.loc[train.index])

                        base_fc = predict_prophet(train['sales_qty'], horizon, freq='D')
                        base_fc = np.clip(base_fc, 0, None)

                        gated_fc, event_score, gate_on = apply_event_gating(
                            base_fc, beta, events.loc[test.index], tau
                        )

                        actual = test['sales_qty'].values
                        wape_base = compute_wape(actual, base_fc)
                        wape_gated = compute_wape(actual, gated_fc)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("기저 예측(Layer 1) WAPE", f"{wape_base:.1f}%")
                        m2.metric("이벤트 게이팅(H1) WAPE", f"{wape_gated:.1f}%",
                                  delta=f"{wape_gated - wape_base:+.1f}%p", delta_color="inverse")
                        m3.metric("게이팅 발동 일수", f"{int(gate_on.sum())} / {horizon}일")

                        if wape_gated < wape_base:
                            st.success(
                                f"H1 지지: '{ev_item}'은 이벤트 게이팅 적용 시 WAPE가 "
                                f"{wape_base - wape_gated:.1f}%p 개선됐습니다."
                            )
                        else:
                            st.warning(
                                f"H1 미지지: '{ev_item}'은 이번 검증 기간에서 이벤트 게이팅이 오히려 "
                                f"WAPE를 {wape_gated - wape_base:.1f}%p 악화시켰습니다."
                            )

                        fig3 = go.Figure()
                        fig3.add_trace(go.Scatter(x=test.index, y=actual, name='실제',
                                                   line=dict(color='black', width=3)))
                        fig3.add_trace(go.Scatter(x=test.index, y=base_fc, name='기저 예측(Layer 1)',
                                                   line=dict(color='blue', dash='dot')))
                        fig3.add_trace(go.Scatter(x=test.index, y=gated_fc, name='이벤트 게이팅 예측(H1)',
                                                   line=dict(color='red', width=2)))
                        for d, on in zip(test.index, gate_on):
                            if on:
                                fig3.add_vrect(x0=d - pd.Timedelta(hours=12), x1=d + pd.Timedelta(hours=12),
                                               fillcolor='orange', opacity=0.15, line_width=0)
                        fig3.update_layout(
                            height=450, hovermode='x unified',
                            title=f"{ev_item} — 이벤트 게이팅 백테스트 (주황 음영 = 게이팅 발동일)"
                        )
                        st.plotly_chart(fig3, use_container_width=True)

                        with st.expander("📊 추정된 이벤트 탄력도(β) 보기"):
                            beta_df = pd.DataFrame({'이벤트': list(beta.keys()), '탄력도(β)': list(beta.values())})
                            beta_df = beta_df.reindex(
                                beta_df['탄력도(β)'].abs().sort_values(ascending=False).index
                            )
                            st.dataframe(beta_df, use_container_width=True, hide_index=True)
                            st.caption("β > 0: 해당 이벤트일 때 판매량 증가 경향 / β < 0: 감소 경향 "
                                       "(학습 구간 기준 선형회귀 추정치, Σβₖ·eₖ가 이벤트 강도 점수)")
            else:
                st.info("👈 방학·공휴일·날씨·시즌 이벤트가 포함된 통합 데이터셋 CSV를 업로드하면 H1 가설을 검증할 수 있습니다.")

        # TAB 4: 품목간 수요연쇄(H2′) 검증
        with tab4:
            st.subheader("🔗 품목간 수요연쇄 (Layer 3, H2′) — 리드-래그 백테스트")
            st.caption(
                "한 품목 카테고리(선행)의 판매 변화가 며칠 뒤 다른 카테고리(후행) 판매에 영향을 주는지 "
                "교차상관·Granger 인과검정으로 확인하고, 그 신호를 반영한 예측이 기저 예측보다 정확한지 검증합니다."
            )
            chain_file = st.file_uploader(
                "이벤트/카테고리 통합 데이터셋 CSV 업로드 (date, category, sales_qty 컬럼 포함)",
                type=['csv'], key='chain_csv'
            )

            if chain_file is not None:
                cdf = load_event_dataset(chain_file)
                required_cols = {'date', 'category', 'sales_qty'}
                missing_cols = required_cols - set(cdf.columns)

                if missing_cols:
                    st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")
                elif not ML_AVAILABLE:
                    st.warning("scikit-learn이 설치되어 있지 않아 계수를 추정할 수 없습니다.")
                else:
                    categories = sorted(cdf['category'].unique())
                    c1, c2 = st.columns(2)
                    with c1:
                        lead_cat = st.selectbox("선행 카테고리 (원인)", categories, index=0, key='lead_cat')
                    with c2:
                        lag_options = [c for c in categories if c != lead_cat]
                        lag_cat = st.selectbox("후행 카테고리 (결과)", lag_options, index=0, key='lag_cat')

                    max_lag = st.slider("탐색할 최대 시차(일)", 3, 21, 14, key='chain_max_lag')
                    horizon2 = st.slider("백테스트 검증 기간(일)", 7, 28, 14, key='chain_horizon')

                    cat_daily = cdf.groupby(['date', 'category'])['sales_qty'].sum().unstack('category').sort_index()
                    cat_daily = cat_daily.asfreq('D').fillna(0.0)

                    if lead_cat not in cat_daily.columns or lag_cat not in cat_daily.columns:
                        st.warning("선택한 카테고리의 일별 데이터가 부족합니다.")
                    elif len(cat_daily) < horizon2 + max_lag + 30:
                        st.warning("검증에 필요한 데이터가 부족합니다.")
                    else:
                        leading = cat_daily[lead_cat]
                        lagging = cat_daily[lag_cat]

                        best_lag, best_corr, corr_by_lag = find_optimal_lag(leading, lagging, 1, max_lag)
                        p_value = granger_pvalue(lagging, leading, best_lag)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("최적 시차 L*", f"{best_lag}일")
                        m2.metric("교차상관 계수", f"{best_corr:.3f}")
                        m3.metric("Granger p-value", f"{p_value:.4f}" if p_value is not None else "계산 불가")

                        if p_value is not None and p_value < 0.05:
                            st.success(
                                f"H2′ 지지: '{lead_cat}' → '{lag_cat}' 시차 {best_lag}일 인과관계가 "
                                f"통계적으로 유의합니다 (p<0.05)."
                            )
                        else:
                            st.warning(
                                f"H2′ 미지지: '{lead_cat}' → '{lag_cat}' 인과관계가 통계적으로 유의하지 않습니다."
                            )

                        lag_df = pd.DataFrame({
                            '시차(일)': list(corr_by_lag.keys()),
                            '상관계수': list(corr_by_lag.values())
                        })
                        fig4a = px.bar(lag_df, x='시차(일)', y='상관계수',
                                        title=f"'{lead_cat}' → '{lag_cat}' 시차별 교차상관")
                        fig4a.add_hline(y=0, line_color='gray')
                        st.plotly_chart(fig4a, use_container_width=True)

                        leading_lagged_full = leading.shift(best_lag)
                        valid_idx = leading_lagged_full.notna()
                        lagging_v = lagging[valid_idx]
                        leading_lagged_v = leading_lagged_full[valid_idx]

                        if len(lagging_v) < horizon2 + 30:
                            st.warning("시차 반영 후 백테스트에 필요한 데이터가 부족합니다.")
                        else:
                            train_lagging = lagging_v.iloc[:-horizon2]
                            test_lagging = lagging_v.iloc[-horizon2:]
                            train_leading_lagged = leading_lagged_v.iloc[:-horizon2]
                            test_leading_lagged = leading_lagged_v.iloc[-horizon2:]

                            base_fc2 = predict_prophet(train_lagging, horizon2, freq='D')
                            base_fc2 = np.clip(base_fc2, 0, None)

                            reg = fit_item_chain_gain(train_lagging, train_leading_lagged)
                            chain_fc = predict_item_chain(base_fc2, reg, test_leading_lagged)

                            actual2 = test_lagging.values
                            wape_base2 = compute_wape(actual2, base_fc2)
                            wape_chain = compute_wape(actual2, chain_fc)

                            m4, m5 = st.columns(2)
                            m4.metric(f"'{lag_cat}' 기저 예측 WAPE", f"{wape_base2:.1f}%")
                            m5.metric("품목간 연쇄 반영 WAPE", f"{wape_chain:.1f}%",
                                      delta=f"{wape_chain - wape_base2:+.1f}%p", delta_color="inverse")

                            fig4b = go.Figure()
                            fig4b.add_trace(go.Scatter(x=test_lagging.index, y=actual2, name='실제',
                                                        line=dict(color='black', width=3)))
                            fig4b.add_trace(go.Scatter(x=test_lagging.index, y=base_fc2, name='기저 예측(Layer 1)',
                                                        line=dict(color='blue', dash='dot')))
                            fig4b.add_trace(go.Scatter(x=test_lagging.index, y=chain_fc,
                                                        name=f"품목간 연쇄 반영({lead_cat}→{lag_cat})",
                                                        line=dict(color='green', width=2)))
                            fig4b.update_layout(
                                height=450, hovermode='x unified',
                                title=f"'{lag_cat}' 예측 백테스트 — 선행: '{lead_cat}' (시차 {best_lag}일)"
                            )
                            st.plotly_chart(fig4b, use_container_width=True)
            else:
                st.info("👈 카테고리별 판매 데이터가 포함된 통합 데이터셋 CSV를 업로드하면 H2′ 가설을 검증할 수 있습니다.")

        # TAB 5: 이상탐지
        with tab5:
            st.subheader("🚨 실시간 이상탐지 (Isolation Forest + 3σ + EWMA)")
            st.caption(
                "일별 판매량의 급감·급증을 세 가지 방법으로 함께 탐지하고, 품목을 선택했을 때는 "
                "전체 매출도 같이 흔들렸는지(공통 요인) 그 품목만 특이했는지를 구분합니다."
            )
            anomaly_file = st.file_uploader(
                "판매 데이터셋 CSV 업로드 (date, item, sales_qty 컬럼 포함)",
                type=['csv'], key='anomaly_csv'
            )

            if anomaly_file is not None:
                adf = load_event_dataset(anomaly_file)
                required_cols = {'date', 'item', 'sales_qty'}
                missing_cols = required_cols - set(adf.columns)

                if missing_cols:
                    st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")
                elif not ML_AVAILABLE:
                    st.warning("scikit-learn이 설치되어 있지 않아 Isolation Forest를 사용할 수 없습니다.")
                else:
                    targets = ['(전체 매출)'] + sorted(adf['item'].unique())
                    target = st.selectbox("탐지 대상", targets, key='anomaly_target')
                    contamination = st.slider(
                        "Isolation Forest 이상치 비율", 0.01, 0.15, 0.05, 0.01, key='anomaly_contam'
                    )

                    daily_total = adf.groupby('date')['sales_qty'].sum().asfreq('D').fillna(0.0)

                    if target == '(전체 매출)':
                        target_series = daily_total
                    else:
                        target_series = adf[adf['item'] == target].groupby('date')['sales_qty'].sum()
                        target_series = target_series.reindex(daily_total.index, fill_value=0.0)

                    if len(target_series) < 40:
                        st.warning("이상탐지에 필요한 데이터(최소 40일)가 부족합니다.")
                    else:
                        sigma_df = rolling_3sigma_flags(target_series)
                        ewma_df = ewma_flags(target_series)
                        iso_flags = isolation_forest_flags(target_series, contamination)

                        combined = pd.DataFrame({
                            '3시그마': sigma_df['is_anomaly'],
                            'EWMA': ewma_df['is_anomaly'],
                            'IsolationForest': iso_flags,
                        })
                        combined['탐지방법수'] = combined.sum(axis=1)
                        combined['이상여부'] = combined['탐지방법수'] >= 2

                        n_anomaly = int(combined['이상여부'].sum())
                        m1, m2 = st.columns(2)
                        m1.metric("탐지된 이상일 (2개 이상 방법 합의)", f"{n_anomaly}일 / {len(combined)}일")
                        m2.metric("Isolation Forest 단독 탐지", f"{int(iso_flags.sum())}일")
                        st.caption(
                            "Isolation Forest는 지정한 비율(위 슬라이더)만큼 '상대적으로' 가장 이상한 날을 "
                            "항상 골라내는 방식이라, 급변이 없는 기간에도 일정 비율은 탐지됩니다."
                        )

                        scope_df = None
                        if target != '(전체 매출)':
                            total_sigma = rolling_3sigma_flags(daily_total)
                            total_ewma = ewma_flags(daily_total)
                            total_flag = total_sigma['is_anomaly'] | total_ewma['is_anomaly']

                            anomaly_dates = combined.index[combined['이상여부']]
                            scope_rows = []
                            for d in anomaly_dates:
                                scope_rows.append({
                                    '날짜': d.date(),
                                    '값': target_series.loc[d],
                                    '구분': '전체 매출 동반 이상' if total_flag.get(d, False) else '품목 단독 이상',
                                })
                            scope_df = pd.DataFrame(scope_rows)
                            if not scope_df.empty:
                                n_shared = int((scope_df['구분'] == '전체 매출 동반 이상').sum())
                                n_solo = int((scope_df['구분'] == '품목 단독 이상').sum())
                                st.info(f"전체 매출 동반 이상 {n_shared}건 / '{target}' 단독 이상 {n_solo}건")

                        fig5 = go.Figure()
                        fig5.add_trace(go.Scatter(x=sigma_df.index, y=sigma_df['ucl'], name='3σ 상한',
                                                   line=dict(color='lightblue', width=1, dash='dot')))
                        fig5.add_trace(go.Scatter(x=sigma_df.index, y=sigma_df['lcl'], name='3σ 하한',
                                                   line=dict(color='lightblue', width=1, dash='dot'),
                                                   fill='tonexty', fillcolor='rgba(173,216,230,0.15)'))
                        fig5.add_trace(go.Scatter(x=sigma_df.index, y=sigma_df['value'], name='실제값',
                                                   line=dict(color='gray', width=1)))
                        anomaly_dates_plot = combined.index[combined['이상여부']]
                        fig5.add_trace(go.Scatter(
                            x=anomaly_dates_plot, y=target_series.loc[anomaly_dates_plot],
                            mode='markers', name='이상치(2개+ 합의)',
                            marker=dict(color='red', size=10, symbol='x')
                        ))
                        fig5.update_layout(
                            height=450, hovermode='x unified',
                            title=f"{target} — 일별 판매량 및 이상탐지 결과"
                        )
                        st.plotly_chart(fig5, use_container_width=True)

                        with st.expander("📋 이상치 상세 목록"):
                            if scope_df is not None:
                                st.dataframe(scope_df, use_container_width=True, hide_index=True)
                            else:
                                detail = combined[combined['이상여부']].copy()
                                detail['값'] = target_series.loc[detail.index]
                                st.dataframe(detail, use_container_width=True)

                        st.markdown("---")
                        if st.button("📤 Grafana용 SQLite로 내보내기", key='export_grafana'):
                            n_rows = export_monitoring_metrics(
                                target, sigma_df, ewma_df, iso_flags, combined, scope_df
                            )
                            st.success(
                                f"{n_rows}일치 지표를 `{MONITORING_DB_PATH.name}`에 저장했습니다 "
                                f"(target='{target}'). Grafana 대시보드에서 새로고침하면 반영됩니다."
                            )
            else:
                st.info("👈 품목별 판매 데이터가 포함된 통합 데이터셋 CSV를 업로드하면 이상탐지를 실행할 수 있습니다.")

        # TAB 6: SHAP 기여도 분석
        with tab6:
            st.subheader("🧮 SHAP 기여도 분석 (XAI)")
            st.caption(
                "예측치를 '기저치 + 이벤트별 기여도'로 분해합니다. 각 예측이 왜 그 값이 나왔는지 "
                "품목·날짜 단위로 근거를 확인할 수 있습니다."
            )
            shap_file = st.file_uploader(
                "이벤트 통합 데이터셋 CSV 업로드 (date, item, sales_qty, 이벤트·기온 컬럼 포함)",
                type=['csv'], key='shap_csv'
            )

            if shap_file is not None:
                sdf = load_event_dataset(shap_file)
                required_cols = {'date', 'item', 'sales_qty', 'is_weekend', 'is_holiday',
                                  'is_vacation', 'season_period', 'precip_type', 'temperature'}
                missing_cols = required_cols - set(sdf.columns)

                if missing_cols:
                    st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")
                elif not ML_AVAILABLE:
                    st.warning("scikit-learn이 설치되어 있지 않아 모델을 학습할 수 없습니다.")
                elif not SHAP_AVAILABLE:
                    st.warning("shap이 설치되어 있지 않습니다. `pip install shap` 후 이용할 수 있습니다.")
                else:
                    shap_items = sorted(sdf['item'].unique())
                    shap_item = st.selectbox("품목 선택", shap_items, key='shap_item')
                    shap_horizon = st.slider("SHAP 분석 대상 기간(최근 N일)", 7, 60, 14, key='shap_horizon')

                    item_daily = sdf[sdf['item'] == shap_item].groupby('date').agg(
                        sales_qty=('sales_qty', 'sum'),
                        is_weekend=('is_weekend', 'first'),
                        is_holiday=('is_holiday', 'first'),
                        is_vacation=('is_vacation', 'first'),
                        season_period=('season_period', 'first'),
                        precip_type=('precip_type', 'first'),
                        temperature=('temperature', 'first'),
                    ).sort_index().asfreq('D')
                    item_daily['sales_qty'] = item_daily['sales_qty'].fillna(0.0)
                    item_daily = item_daily.ffill()

                    if len(item_daily) < shap_horizon + 60:
                        st.warning("모델 학습에 필요한 데이터(최소 60일 + 분석기간)가 부족합니다.")
                    else:
                        X = build_shap_features(item_daily)
                        y = item_daily['sales_qty']

                        X_train, X_test = X.iloc[:-shap_horizon], X.iloc[-shap_horizon:]
                        y_train, y_test = y.iloc[:-shap_horizon], y.iloc[-shap_horizon:]

                        with st.spinner("모델 학습 및 SHAP 값 계산 중..."):
                            model = train_shap_model(X_train, y_train)
                            shap_values, base_value = explain_with_shap(model, X_test)
                            pred_test = model.predict(X_test)

                        st.markdown("#### 전역 특징 중요도 (평균 |SHAP|)")
                        importance = pd.Series(
                            np.abs(shap_values).mean(axis=0), index=X.columns
                        ).sort_values(ascending=True)
                        importance.index = [SHAP_LABELS.get(c, c) for c in importance.index]
                        fig6a = px.bar(
                            importance, orientation='h',
                            labels={'value': '평균 |SHAP| (판매량 영향력)', 'index': '피처'},
                            title=f"'{shap_item}' 예측에 대한 피처별 평균 기여도"
                        )
                        st.plotly_chart(fig6a, use_container_width=True)

                        st.markdown("#### 날짜별 예측 분해")
                        sel_date = st.selectbox(
                            "날짜 선택", list(X_test.index), key='shap_date',
                            format_func=lambda d: d.strftime('%Y-%m-%d (%a)')
                        )
                        idx = list(X_test.index).index(sel_date)
                        contribs = pd.Series(shap_values[idx], index=X.columns)
                        pred = pred_test[idx]
                        actual = y_test.iloc[idx]

                        briefing = format_shap_briefing(
                            shap_item, sel_date.strftime('%Y-%m-%d'), base_value, contribs, pred, actual
                        )
                        st.chat_message("assistant").write(briefing)

                        contrib_df = pd.DataFrame({
                            '피처': [SHAP_LABELS.get(c, c) for c in contribs.index],
                            'SHAP 기여도': contribs.values,
                        })
                        contrib_df = contrib_df.reindex(
                            contrib_df['SHAP 기여도'].abs().sort_values(ascending=False).index
                        )
                        fig6b = px.bar(
                            contrib_df, x='SHAP 기여도', y='피처', orientation='h',
                            color='SHAP 기여도', color_continuous_scale='RdBu', color_continuous_midpoint=0,
                            title=f"{sel_date.strftime('%Y-%m-%d')} 예측 기여도 분해 (기저 {base_value:.1f}개)"
                        )
                        st.plotly_chart(fig6b, use_container_width=True)

                        with st.expander("📋 기여도 상세 표"):
                            st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                            st.caption(
                                f"검증: 기저치({base_value:.1f}) + 전체 기여도 합({contribs.sum():.1f}) "
                                f"= 모델 예측치({pred:.1f})"
                            )
            else:
                st.info("👈 이벤트 통합 데이터셋 CSV를 업로드하면 SHAP 기여도 분석을 실행할 수 있습니다.")

        # TAB 7: What-if 채팅
        with tab7:
            st.subheader("💬 What-if 채팅 — 이벤트 조건별 예상 판매량")
            st.caption(
                "이벤트 조건(비·눈·공휴일·방학·시즌)을 체크하거나 채팅으로 언급하면, "
                "이벤트 우선 게이팅(H1) 모델로 실제 계산한 예상 판매량과 권장 생산량을 답합니다. "
                "※ 채팅 문구는 키워드 매칭으로 조건을 인식합니다(LLM 미연동, 계산값은 실제 모델 결과)."
            )
            whatif_file = st.file_uploader(
                "이벤트 통합 데이터셋 CSV 업로드 (date, item, sales_qty, 이벤트 컬럼 포함)",
                type=['csv'], key='whatif_csv'
            )

            if whatif_file is not None:
                wdf = load_event_dataset(whatif_file)
                required_cols = {'date', 'item', 'sales_qty', 'is_weekend', 'is_holiday',
                                  'is_vacation', 'season_period', 'precip_type'}
                missing_cols = required_cols - set(wdf.columns)

                if missing_cols:
                    st.error(f"필수 컬럼이 없습니다: {', '.join(sorted(missing_cols))}")
                elif not ML_AVAILABLE:
                    st.warning("scikit-learn이 설치되어 있지 않아 이벤트 탄력도를 추정할 수 없습니다.")
                else:
                    wi_item = st.selectbox("대상 품목", sorted(wdf['item'].unique()), key='whatif_item')

                    item_daily = wdf[wdf['item'] == wi_item].groupby('date').agg(
                        sales_qty=('sales_qty', 'sum'),
                        is_weekend=('is_weekend', 'first'),
                        is_holiday=('is_holiday', 'first'),
                        is_vacation=('is_vacation', 'first'),
                        season_period=('season_period', 'first'),
                        precip_type=('precip_type', 'first'),
                    ).sort_index().asfreq('D')
                    item_daily['sales_qty'] = item_daily['sales_qty'].fillna(0.0)
                    item_daily = item_daily.ffill()

                    if len(item_daily) < 30:
                        st.warning("모델 학습에 필요한 데이터(최소 30일)가 부족합니다.")
                    else:
                        events_full = build_event_dummies(item_daily)
                        beta = fit_event_elasticity(item_daily['sales_qty'], events_full)
                        base_forecast = float(np.clip(
                            predict_holt_trend(item_daily['sales_qty'].values, 1)[0], 0, None
                        ))

                        st.markdown("##### 🔧 빠른 이벤트 선택 (채팅 대신 체크해도 됨)")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            chk_rain = st.checkbox("🌧 비", key='wi_rain')
                            chk_snow = st.checkbox("❄ 눈", key='wi_snow')
                        with c2:
                            chk_holiday = st.checkbox("🎌 공휴일", key='wi_holiday')
                            chk_weekend = st.checkbox("📅 주말", key='wi_weekend')
                        with c3:
                            chk_vacation = st.checkbox("🏫 방학", key='wi_vacation')
                        with c4:
                            season_choice = st.selectbox(
                                "시즌", ['없음', '크리스마스', '수능', '밸런타인', '추석'], key='wi_season'
                            )
                        tau_wi = st.slider("게이팅 임계치 τ", 0.0, 0.3, 0.10, 0.01, key='wi_tau')

                        if 'whatif_messages' not in st.session_state:
                            st.session_state.whatif_messages = []

                        if st.button("🗑 대화 초기화", key='whatif_clear'):
                            st.session_state.whatif_messages = []
                            st.rerun()

                        for msg in st.session_state.whatif_messages:
                            st.chat_message(msg['role']).write(msg['content'])

                        user_text = st.chat_input(
                            f"'{wi_item}'에 대해 물어보세요 (예: 내일 비 오고 방학이면?)"
                        )
                        if user_text:
                            st.session_state.whatif_messages.append({'role': 'user', 'content': user_text})
                            text_flags = parse_whatif_keywords(user_text)

                            events = {
                                'is_weekend': float(chk_weekend or text_flags['is_weekend']),
                                'is_holiday': float(chk_holiday or text_flags['is_holiday']),
                                'is_vacation': float(chk_vacation or text_flags['is_vacation']),
                                'is_rain': float(chk_rain or text_flags['is_rain']),
                                'is_snow': float(chk_snow or text_flags['is_snow']),
                                'is_christmas': float(season_choice == '크리스마스' or text_flags['is_christmas']),
                                'is_suneung': float(season_choice == '수능' or text_flags['is_suneung']),
                                'is_valentine': float(season_choice == '밸런타인' or text_flags['is_valentine']),
                                'is_chuseok': float(season_choice == '추석' or text_flags['is_chuseok']),
                            }

                            event_score = sum(beta[k] * v for k, v in events.items())
                            gate_on = abs(event_score) >= tau_wi
                            final_forecast = base_forecast * (1 + event_score) if gate_on else base_forecast
                            final_forecast = float(np.clip(final_forecast, 0, None))
                            production_qty = int(np.ceil(final_forecast * 1.1))

                            active = [SHAP_LABELS.get(k, k) for k, v in events.items() if v]
                            cond_str = ', '.join(active) if active else '특별한 이벤트 없음'
                            pct = (
                                (final_forecast - base_forecast) / base_forecast * 100
                                if base_forecast > 0 else 0.0
                            )

                            reply = (
                                f"조건({cond_str})을 반영하면 '{wi_item}' 예상 판매량은 "
                                f"{base_forecast:.0f}개 → **{final_forecast:.0f}개** ({pct:+.1f}%)입니다. "
                                f"안전재고 10%를 더한 권장 생산량은 **{production_qty}개**입니다."
                            )
                            if not gate_on:
                                reply += (
                                    f" (이벤트 강도 {event_score:+.2f}가 임계치 τ={tau_wi:.2f} 미만이라 "
                                    f"게이팅 미발동, 기저 예측을 유지했습니다.)"
                                )

                            st.session_state.whatif_messages.append({'role': 'assistant', 'content': reply})
                            st.rerun()
            else:
                st.info("👈 이벤트 통합 데이터셋 CSV를 업로드하면 What-if 채팅을 사용할 수 있습니다.")

else:
    st.info("👈 왼쪽 사이드바에서 빵집 판매내역 CSV 파일을 업로드해주세요.")
