"""
========================================================================
 Crypto ML Predictor — Advanced Cryptocurrency Prediction Script
========================================================================
مصدر البيانات: Binance (via ccxt)
يدعم: أي عملة / أي فريم زمني / حتى 5 سنين بيانات تاريخية
الموديل: LightGBM + XGBoost (Ensemble) + مقارنة بأداء Random/Baseline
Backtesting: Walk-Forward مع تحديد تاريخ بداية/نهاية الاختبار من شاشة السكربت

تحذير مهم (اقرأه قبل الاستخدام):
لا يوجد نموذج تعلم آلي يمكنه التنبؤ بدقة عالية جداً بأسعار العملات الرقمية
بشكل مستمر، لأن السوق يتأثر بعوامل لا يمكن التنبؤ بها (أخبار، قرارات،
سيولة، تلاعب). هذا السكربت يعطيك بنية قوية ومنهجية علمية صحيحة
(بدون تسريب بيانات من المستقبل)، لكنه لا يضمن أرباحاً حقيقية.
استخدمه للتعليم والبحث فقط، وليس كنصيحة استثمارية.
========================================================================
"""

import os
import sys
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
# 1) إعدادات المستخدم — عدّل هنا فقط
# ------------------------------------------------------------------------
CONFIG = {
    "SYMBOL": "TRX/USDT",           # مخصّص لعملة TRON (TRX)
    "TIMEFRAME": "8h",              # مؤكد ومثبّت كأساس للسكربت كله (تقدر تغيّره لو حبيت)
    "YEARS_BACK": 5,               # عدد سنين البيانات للتعلم (حتى 5 سنين)

    # -------- شاشة تحديد التاريخ لعمل Backtest على فترة قديمة محددة --------
    # اتركهم None لاستخدام كل البيانات المتاحة تلقائياً بحساب YEARS_BACK
    "TEST_START_DATE": None,       # مثال: "2023-01-01"
    "TEST_END_DATE": None,         # مثال: "2023-06-01"

    # -------- إعدادات التنبؤ --------
    "PREDICTION_HORIZON": 1,       # التنبؤ بعد كام شمعة (1 = الشمعة الجاية)
    "TARGET_TYPE": "direction",    # "direction" (صعود/نزول) أو "price" (سعر فعلي)
    "MIN_MOVE_THRESHOLD": 0.002,    # (Dead Zone) استبعاد الحركات الأصغر من 0.2% كنويز — صفر = تعطيل

    # -------- تحسينات الذكاء --------
    "USE_HIGHER_TIMEFRAME_TREND": True,   # إضافة اتجاه فريم زمني أعلى (يومي) كـ feature
    "HIGHER_TIMEFRAME": "1d",
    "USE_CROSS_ASSET_FEATURES": True,     # إضافة حركة BTC/ETH كـ features (مهم جداً للعملات البديلة زي TRX)
    "CROSS_ASSETS": ["BTC/USDT", "ETH/USDT"],
    "USE_WALK_FORWARD": True,             # تقييم بـ Walk-Forward بدل split واحد فقط
    "WALK_FORWARD_SPLITS": 5,

    # -------- إعدادات الموديل --------
    "MODELS_TO_USE": ["lightgbm", "xgboost"],  # يمكن استخدام واحد أو الاثنين معاً (Ensemble)
    "TRAIN_TEST_SPLIT_RATIO": 0.85,             # نسبة بيانات التدريب (الباقي = اختبار زمني حقيقي)

    "OUTPUT_DIR": "/mnt/user-data/outputs",
}

# ------------------------------------------------------------------------
# 2) تثبيت المكتبات المطلوبة تلقائياً إذا لم تكن موجودة
# ------------------------------------------------------------------------
def ensure_packages():
    required = {
        "ccxt": "ccxt",
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "matplotlib": "matplotlib",
        "ta": "ta",
        "plotly": "plotly",
        "streamlit": "streamlit",
        "requests": "requests",
        "scipy": "scipy",
    }
    import importlib
    for mod, pip_name in required.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            os.system(f"{sys.executable} -m pip install --break-system-packages -q {pip_name}")


ensure_packages()

import ccxt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, mean_absolute_error, roc_auc_score)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None


# ------------------------------------------------------------------------
# 3) جلب البيانات من Binance (بالتقسيم على دفعات لتغطية سنوات كاملة)
# ------------------------------------------------------------------------
def fetch_binance_ohlcv(symbol, timeframe, years_back):
    print(f"[+] جاري تحميل بيانات {symbol} فريم {timeframe} لآخر {years_back} سنين من Binance...")
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",
            "fetchMarkets": ["spot"],  # يمنع محاولة تحميل بيانات العقود الآجلة (Futures) المحظورة جغرافياً
            "fetchCurrencies": False,
        },
    })

    # مهم جداً لو السكربت شغال على سيرفر سحابي (زي Streamlit Cloud): بعض
    # سيرفرات Binance الرئيسية بترفض الاتصال من مواقع/IP معينة (خطأ 451).
    # data-api.binance.vision هو مرايا عامة رسمية من Binance لبيانات السوق
    # فقط (بدون تداول فعلي)، وغير خاضع لنفس القيود الجغرافية — نستخدمه
    # كمصدر أساسي أكثر ثباتاً على السحابة.
    # (بنستبدل الدومين بس، ونحافظ على باقي مسار الـ API زي ما هو عشان
    # نضمن التوافق مع أي إصدار ccxt، بدل ما نكتب المسار يدوي وممكن نغلط فيه)
    try:
        original_public_url = exchange.urls["api"]["public"]
        exchange.urls["api"]["public"] = original_public_url.replace(
            "api.binance.com", "data-api.binance.vision"
        )
        print(f"[+] مصدر البيانات: {exchange.urls['api']['public']}")
    except Exception:
        pass  # لو تغيّرت بنية ccxt الداخلية، نكمل بالمصدر الأساسي بدون كسر السكربت

    ms_per_year = 365 * 24 * 60 * 60 * 1000
    since = exchange.milliseconds() - int(years_back * ms_per_year)

    all_ohlcv = []
    limit = 1000
    max_iterations = 5000  # حماية من اللوب اللانهائي
    iteration = 0
    consecutive_failures = 0
    max_consecutive_failures = 5  # بعد كذا فشل متتالي، نوقف ونبلّغ بخطأ واضح بدل الانتظار للأبد

    while iteration < max_iterations:
        iteration += 1
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            print(f"    [!] خطأ في الجلب ({consecutive_failures}/{max_consecutive_failures}): {e}")
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"تعذّر الاتصال بـ Binance بعد {max_consecutive_failures} محاولات متتالية. "
                    f"آخر خطأ: {e}. لو السكربت شغال على سيرفر سحابي، من المحتمل إن Binance حاظر "
                    f"الاتصال من موقع السيرفر — جرّب لاحقاً أو من سيرفر/شبكة مختلفة."
                )
            time.sleep(3)
            continue

        if not batch:
            break

        all_ohlcv.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1

        if last_ts >= exchange.milliseconds() - 60_000:
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_ohlcv:
        raise RuntimeError(
            "لم يتم استلام أي بيانات من Binance. تأكد من اسم العملة والفريم الزمني، "
            "أو إن الاتصال بـ Binance غير محظور من موقع السيرفر الحالي."
        )

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.drop_duplicates(subset="timestamp", inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[+] تم تحميل {len(df)} شمعة — من {df['timestamp'].min()} إلى {df['timestamp'].max()}")
    return df


# ------------------------------------------------------------------------
# 4) بناء المؤشرات الفنية (Feature Engineering) — بدون أي تسريب من المستقبل
# ------------------------------------------------------------------------
def build_features(df):
    print("[+] جاري بناء المؤشرات الفنية...")
    data = df.copy()

    # --- مؤشرات الاتجاه والمتوسطات ---
    for period in [5, 10, 20, 50, 100, 200]:
        data[f"ema_{period}"] = ta.trend.ema_indicator(data["close"], window=period)
        data[f"sma_{period}"] = ta.trend.sma_indicator(data["close"], window=period)

    # --- MACD ---
    macd = ta.trend.MACD(data["close"])
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_diff"] = macd.macd_diff()

    # --- RSI ---
    data["rsi_14"] = ta.momentum.rsi(data["close"], window=14)
    data["rsi_7"] = ta.momentum.rsi(data["close"], window=7)

    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(data["close"], window=20)
    data["bb_high"] = bb.bollinger_hband()
    data["bb_low"] = bb.bollinger_lband()
    data["bb_width"] = bb.bollinger_wband()
    data["bb_pct"] = bb.bollinger_pband()

    # --- ATR (تقلب السوق) ---
    data["atr_14"] = ta.volatility.average_true_range(data["high"], data["low"], data["close"], window=14)

    # --- Stochastic Oscillator ---
    stoch = ta.momentum.StochasticOscillator(data["high"], data["low"], data["close"])
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()

    # --- ADX (قوة الترند) ---
    data["adx"] = ta.trend.adx(data["high"], data["low"], data["close"], window=14)

    # --- مؤشرات إضافية أقوى (Momentum/Trend) ---
    data["cci_20"] = ta.trend.cci(data["high"], data["low"], data["close"], window=20)
    data["williams_r"] = ta.momentum.williams_r(data["high"], data["low"], data["close"], lbp=14)
    data["roc_10"] = ta.momentum.roc(data["close"], window=10)
    data["mfi_14"] = ta.volume.money_flow_index(data["high"], data["low"], data["close"], data["volume"], window=14)

    # --- Ichimoku Cloud (يعتبر من أقوى مؤشرات تحديد الاتجاه العام) ---
    ichi = ta.trend.IchimokuIndicator(data["high"], data["low"], window1=9, window2=26, window3=52)
    data["ichimoku_a"] = ichi.ichimoku_a()
    data["ichimoku_b"] = ichi.ichimoku_b()
    data["ichimoku_base"] = ichi.ichimoku_base_line()
    data["ichimoku_conv"] = ichi.ichimoku_conversion_line()
    data["price_vs_cloud"] = data["close"] - ((data["ichimoku_a"] + data["ichimoku_b"]) / 2)

    # --- OBV و مؤشرات الحجم ---
    data["obv"] = ta.volume.on_balance_volume(data["close"], data["volume"])
    data["volume_sma_20"] = data["volume"].rolling(20).mean()
    data["volume_change"] = data["volume"].pct_change()

    # --- عوائد سعرية بفترات مختلفة (Lag Features) ---
    for lag in [1, 2, 3, 5, 10, 20]:
        data[f"return_{lag}"] = data["close"].pct_change(lag)

    # --- تقلب السعر (Rolling volatility) ---
    for window in [7, 14, 30]:
        data[f"volatility_{window}"] = data["close"].pct_change().rolling(window).std()

    # --- نسبة السعر لأعلى/أقل سعر خلال فترة ---
    for window in [20, 50]:
        data[f"dist_high_{window}"] = data["close"] / data["high"].rolling(window).max() - 1
        data[f"dist_low_{window}"] = data["close"] / data["low"].rolling(window).min() - 1

    # --- خصائص وقتية (موسمية) ---
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek

    print(f"[+] تم بناء {data.shape[1]} خاصية/مؤشر")
    return data


def add_higher_timeframe_trend(data, symbol, base_timeframe, years_back, higher_timeframe="1d"):
    """
    يضيف اتجاه فريم زمني أعلى (مثلاً يومي) كـ feature — عشان الموديل يعرف
    الاتجاه العام للسوق وما يتصرف عكسه على الفريم الصغير.
    """
    print(f"[+] جاري إضافة اتجاه الفريم الزمني الأعلى ({higher_timeframe})...")
    try:
        higher_df = fetch_binance_ohlcv(symbol, higher_timeframe, years_back)
        higher_df["ema_50_h"] = ta.trend.ema_indicator(higher_df["close"], window=50)
        higher_df["ema_200_h"] = ta.trend.ema_indicator(higher_df["close"], window=200)
        higher_df["htf_trend"] = (higher_df["ema_50_h"] > higher_df["ema_200_h"]).astype(int)
        higher_df["htf_rsi"] = ta.momentum.rsi(higher_df["close"], window=14)

        merge_cols = higher_df[["timestamp", "htf_trend", "htf_rsi"]].copy()
        merge_cols = merge_cols.sort_values("timestamp")

        # ننسب كل صف في البيانات الأساسية لآخر قيمة معروفة من الفريم الأعلى (بدون تسريب مستقبلي)
        data = data.sort_values("timestamp")
        merged = pd.merge_asof(data, merge_cols, on="timestamp", direction="backward")
        print(f"[+] تم دمج اتجاه {higher_timeframe} بنجاح")
        return merged
    except Exception as e:
        print(f"[!] تعذّرت إضافة اتجاه الفريم الأعلى: {e} — سيتم التخطي")
        data["htf_trend"] = np.nan
        data["htf_rsi"] = np.nan
        return data


def add_cross_asset_features(data, timeframe, years_back, cross_assets=("BTC/USDT", "ETH/USDT")):
    """
    يضيف حركة الأصول المرتبطة (BTC, ETH) كـ features — لأن العملات البديلة
    زي TRX غالباً بتتحرك 'تبع' البيتكوين، والموديل محتاج يعرف ده صريح.
    كمان بيحسب معامل الارتباط المتحرك (Rolling Correlation) بين العملة الأساسية
    والبيتكوين، عشان يوريك إمتى العملة بتاخد قرارها لوحدها وإمتى بتتبع السوق.
    """
    print(f"[+] جاري إضافة ميزات الأصول المرتبطة ({', '.join(cross_assets)})...")
    data = data.sort_values("timestamp").reset_index(drop=True)

    for asset in cross_assets:
        try:
            asset_tag = asset.split("/")[0].lower()  # مثال: btc, eth
            asset_df = fetch_binance_ohlcv(asset, timeframe, years_back)
            asset_df[f"{asset_tag}_return_1"] = asset_df["close"].pct_change(1)
            asset_df[f"{asset_tag}_return_5"] = asset_df["close"].pct_change(5)
            asset_df[f"{asset_tag}_rsi_14"] = ta.momentum.rsi(asset_df["close"], window=14)
            asset_df[f"{asset_tag}_ema_trend"] = (
                ta.trend.ema_indicator(asset_df["close"], window=20)
                > ta.trend.ema_indicator(asset_df["close"], window=50)
            ).astype(int)

            merge_cols = asset_df[[
                "timestamp", f"{asset_tag}_return_1", f"{asset_tag}_return_5",
                f"{asset_tag}_rsi_14", f"{asset_tag}_ema_trend",
            ]].sort_values("timestamp")

            data = pd.merge_asof(data, merge_cols, on="timestamp", direction="backward")

            # معامل الارتباط المتحرك بين عائد العملة الأساسية وعائد الأصل ده (نافذة 30 شمعة)
            if "return_1" in data.columns:
                pass  # سيتم حسابه بعد التأكد من وجود عمود return_1 الأساسي بالأسفل
        except Exception as e:
            print(f"[!] تعذّرت إضافة ميزات {asset}: {e} — سيتم التخطي")

    return data


def add_rolling_correlation_with_btc(data, window=30):
    """يحسب معامل الارتباط المتحرك بين عائد العملة الأساسية وعائد البيتكوين"""
    if "return_1" in data.columns and "btc_return_1" in data.columns:
        data["corr_with_btc_30"] = data["return_1"].rolling(window).corr(data["btc_return_1"])
    return data


def add_tron_onchain_features(data, symbol):
    """
    (اختياري - Best Effort) يحاول جلب بيانات حقيقية عن نشاط شبكة TRON
    (عدد المعاملات اليومية) من Tronscan API العام. لو الطلب فشل (تغيّر API،
    مفيش إنترنت، Rate limit) يتخطى بأمان بدون ما يوقف باقي السكربت.
    """
    if "TRX" not in symbol.upper():
        return data

    print("[+] جاري محاولة جلب بيانات نشاط شبكة TRON (Tronscan)...")
    try:
        import requests
        resp = requests.get(
            "https://apilist.tronscanapi.com/api/system/status",
            timeout=8,
        )
        if resp.status_code == 200:
            print("[+] تم الاتصال بـ Tronscan API بنجاح (بيانات ثابتة إضافية غير مدمجة تلقائياً في الفيتشرز الزمنية)")
        else:
            print(f"[!] Tronscan API رجّع status {resp.status_code} — سيتم التخطي")
    except Exception as e:
        print(f"[!] تعذّر الاتصال بـ Tronscan API ({e}) — سيتم التخطي بأمان، السكربت شغال عادي بباقي الفيتشرز")

    return data


# ------------------------------------------------------------------------
# 4.4) القمم والقيعان (Swing Highs/Lows) — بدون تسريب مستقبلي
# ------------------------------------------------------------------------
def add_swing_points(data, order=5):
    """
    يكشف القمم والقيعان الحقيقية (Swing Highs/Lows) بمنطق الفراكتال:
    نقطة تعتبر قمة لو أعلى من `order` شمعة قبلها وبعدها.
    مهم جداً: النقطة مش "معروفة" إلا بعد ما تعدي `order` شمعة عليها (لازم تشوف
    المستقبل عشان تأكدها)، فبنعمل shift(order) عشان الموديل ميعرفش عنها إلا
    وقتها بالظبط زي ما هيحصل فعلياً وقت التداول الحقيقي (بدون تسريب مستقبلي).
    """
    from scipy.signal import argrelextrema

    print(f"[+] جاري كشف القمم والقيعان (Swing Points, order={order})...")
    data = data.copy()
    highs = data["high"].values
    lows = data["low"].values

    swing_high_idx = argrelextrema(highs, np.greater_equal, order=order)[0]
    swing_low_idx = argrelextrema(lows, np.less_equal, order=order)[0]

    data["is_swing_high"] = 0
    data["is_swing_low"] = 0
    data.loc[data.index[swing_high_idx], "is_swing_high"] = 1
    data.loc[data.index[swing_low_idx], "is_swing_low"] = 1

    # تأخير الإشارة بمقدار order عشان نضمن إننا ماعرفناها إلا بعد تأكدها فعلياً
    data["confirmed_swing_high"] = data["is_swing_high"].shift(order).fillna(0)
    data["confirmed_swing_low"] = data["is_swing_low"].shift(order).fillna(0)

    # آخر قمة/قاعة مؤكدة وسعرها، وبُعد السعر الحالي عنها % (زي مستويات دعم/مقاومة)
    last_swing_high_price = data["high"].where(data["confirmed_swing_high"] == 1).ffill()
    last_swing_low_price = data["low"].where(data["confirmed_swing_low"] == 1).ffill()
    data["dist_to_last_swing_high"] = (data["close"] / last_swing_high_price) - 1
    data["dist_to_last_swing_low"] = (data["close"] / last_swing_low_price) - 1

    # عدد الشموع من آخر قمة/قاعة مؤكدة (يوريك هل السوق في حالة تذبذب أو ترند مستمر)
    data["bars_since_swing_high"] = data.groupby(
        (data["confirmed_swing_high"] == 1).cumsum()
    ).cumcount() if hasattr(data.groupby((data["confirmed_swing_high"] == 1).cumsum()), "cumcount") else np.nan

    print(f"[+] تم كشف {swing_high_idx.size} قمة و {swing_low_idx.size} قاعدة (قبل تأخير التأكيد)")
    # إزالة عمود مؤقت غير مستخدم (احتياطي لو فشلت cumcount)
    data.drop(columns=[c for c in ["bars_since_swing_high"] if data[c].isna().all()], errors="ignore", inplace=True)
    return data


# ------------------------------------------------------------------------
# 4.45) أنماط الشموع اليابانية (Candlestick Patterns) — بدون مكتبات خارجية ثقيلة
# ------------------------------------------------------------------------
def add_candlestick_patterns(data):
    """
    يكشف أشهر أنماط الشموع اليابانية يدوياً (بدون الاعتماد على TA-Lib، اللي
    بيسبب مشاكل تثبيت كتير على ويندوز)، بناءً على شكل كل شمعة ومقارنتها بالسابقة.
    """
    print("[+] جاري كشف أنماط الشموع اليابانية (Engulfing, Doji, Hammer...)...")
    data = data.copy()

    body = (data["close"] - data["open"]).abs()
    candle_range = (data["high"] - data["low"]).replace(0, np.nan)
    upper_wick = data["high"] - data[["open", "close"]].max(axis=1)
    lower_wick = data[["open", "close"]].min(axis=1) - data["low"]
    is_bullish = data["close"] > data["open"]
    is_bearish = data["close"] < data["open"]

    # --- Doji: الجسم صغير جداً مقارنة بمدى الشمعة ---
    data["pattern_doji"] = ((body / candle_range) < 0.1).astype(int)

    # --- Marubozu: جسم كبير بدون فتائل تقريباً (قوة اتجاه واضحة) ---
    data["pattern_marubozu_bull"] = (
        is_bullish & ((body / candle_range) > 0.9)
    ).astype(int)
    data["pattern_marubozu_bear"] = (
        is_bearish & ((body / candle_range) > 0.9)
    ).astype(int)

    # --- Hammer: فتيل سفلي طويل + جسم صغير في الأعلى (انعكاس صعودي محتمل) ---
    data["pattern_hammer"] = (
        (lower_wick > body * 2) & (upper_wick < body * 0.5) & ((body / candle_range) < 0.4)
    ).astype(int)

    # --- Shooting Star: فتيل علوي طويل + جسم صغير في الأسفل (انعكاس نزولي محتمل) ---
    data["pattern_shooting_star"] = (
        (upper_wick > body * 2) & (lower_wick < body * 0.5) & ((body / candle_range) < 0.4)
    ).astype(int)

    # --- Bullish/Bearish Engulfing: الشمعة الحالية "تلتهم" جسم الشمعة السابقة بالكامل ---
    prev_open = data["open"].shift(1)
    prev_close = data["close"].shift(1)
    prev_is_bearish = prev_close < prev_open

    data["pattern_bullish_engulfing"] = (
        is_bullish & prev_is_bearish &
        (data["open"] <= prev_close) & (data["close"] >= prev_open)
    ).astype(int)

    prev_is_bullish = prev_close > prev_open
    data["pattern_bearish_engulfing"] = (
        is_bearish & prev_is_bullish &
        (data["open"] >= prev_close) & (data["close"] <= prev_open)
    ).astype(int)

    pattern_cols = [c for c in data.columns if c.startswith("pattern_")]
    n_detected = int(data[pattern_cols].sum().sum())
    print(f"[+] تم كشف {n_detected} حالة نمط شمعة عبر كل البيانات")
    return data


# ------------------------------------------------------------------------
# 4.46) مراقبة السيولة (Liquidity Monitoring)
# ------------------------------------------------------------------------
def add_liquidity_features(data, window=20):
    """
    يضيف مقاييس سيولة حقيقية مبنية على الحجم والسعر:
    - Volume Z-Score: هل الحجم الحالي أعلى/أقل من الطبيعي بشكل ملحوظ (نشاط غير عادي)
    - Amihud Illiquidity: مقياس معروف في التمويل لقياس "صعوبة تحريك السعر" —
      كل ما زاد، كل ما كانت السيولة أضعف (حركة سعر كبيرة بحجم تداول صغير)
    - الانحراف عن VWAP: هل السعر الحالي أعلى أو أقل من متوسط السعر المرجّح بالحجم
    """
    print("[+] جاري حساب مقاييس السيولة (Volume Z-Score, Amihud, VWAP)...")
    data = data.copy()

    vol_mean = data["volume"].rolling(window).mean()
    vol_std = data["volume"].rolling(window).std()
    data["volume_zscore"] = (data["volume"] - vol_mean) / vol_std.replace(0, np.nan)
    data["is_volume_spike"] = (data["volume_zscore"] > 2).astype(int)
    data["is_liquidity_dry"] = (data["volume_zscore"] < -1.5).astype(int)

    # Amihud Illiquidity Measure = |return| / (volume × close) — نسخة مبسطة شائعة الاستخدام
    ret = data["close"].pct_change().abs()
    dollar_volume = (data["volume"] * data["close"]).replace(0, np.nan)
    data["amihud_illiquidity"] = (ret / dollar_volume) * 1e9  # تكبير رقمي للقراءة بسهولة
    data["amihud_illiquidity_ma"] = data["amihud_illiquidity"].rolling(window).mean()

    # VWAP متحرك (تقريبي على نافذة، وليس VWAP اليومي التقليدي) + الانحراف عنه
    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    vwap = (typical_price * data["volume"]).rolling(window).sum() / data["volume"].rolling(window).sum()
    data["vwap"] = vwap
    data["dist_from_vwap_pct"] = (data["close"] - vwap) / vwap

    print("[+] تم حساب مقاييس السيولة بنجاح")
    return data


def fetch_live_orderbook_snapshot(symbol, depth=20):
    """
    (لحظي فقط - Best Effort) يجيب "لقطة" من دفتر الطلبات الحالي على Binance
    عشان يوريك السيولة الفعلية الآن (مش تاريخية، لأن بيانات الأوردر بوك
    التاريخية غير متاحة مجاناً). يحسب: أفضل Bid/Ask، الفرق بينهم (Spread)،
    وعدم توازن السيولة بين طلبات الشراء والبيع (Order Book Imbalance).
    """
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        ob = exchange.fetch_order_book(symbol, limit=depth)
        best_bid = ob["bids"][0][0] if ob["bids"] else None
        best_ask = ob["asks"][0][0] if ob["asks"] else None
        bid_volume = sum(b[1] for b in ob["bids"])
        ask_volume = sum(a[1] for a in ob["asks"])
        total_vol = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_vol if total_vol > 0 else 0

        spread_pct = ((best_ask - best_bid) / best_bid * 100) if (best_bid and best_ask) else None
        return {
            "best_bid": best_bid, "best_ask": best_ask, "spread_pct": spread_pct,
            "bid_volume": bid_volume, "ask_volume": ask_volume, "imbalance": imbalance,
        }
    except Exception as e:
        print(f"[!] تعذّر جلب دفتر الطلبات اللحظي: {e}")
        return None


# ------------------------------------------------------------------------
# 4.5) جلب الأخبار المؤثرة على TRX/Tron من مصدر مجاني (CryptoCompare News API)
# ------------------------------------------------------------------------
def fetch_trx_news(limit=10):
    """
    يجيب آخر الأخبار المتعلقة بـ TRX/Tron من CryptoCompare News API (مجاني، بدون مفتاح).
    لو فشل الطلب لأي سبب (مفيش إنترنت، تغيّر الـ API، Rate limit)، يرجع قايمة فاضية
    بأمان بدون ما يوقف باقي السكربت.
    """
    print("[+] جاري جلب آخر أخبار TRX/Tron...")
    try:
        import requests
        resp = requests.get(
            "https://min-api.cryptocompare.com/data/v2/news/",
            params={"lang": "EN", "categories": "TRX", "sortOrder": "latest"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        articles = data.get("Data", [])

        if not articles:
            # fallback: أخبار عامة عن الكريبتو وفلترة أي حاجة فيها Tron/TRX في العنوان
            resp2 = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/",
                params={"lang": "EN", "sortOrder": "latest"},
                timeout=10,
            )
            resp2.raise_for_status()
            all_articles = resp2.json().get("Data", [])
            articles = [
                a for a in all_articles
                if "tron" in a.get("title", "").lower() or "trx" in a.get("title", "").lower()
            ]

        news_list = []
        for a in articles[:limit]:
            news_list.append({
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "source": a.get("source_info", {}).get("name", a.get("source", "")),
                "published_on": pd.to_datetime(a.get("published_on", 0), unit="s", utc=True),
                "body": (a.get("body", "") or "")[:200],
            })
        print(f"[+] تم جلب {len(news_list)} خبر")
        return news_list
    except Exception as e:
        print(f"[!] تعذّر جلب الأخبار: {e} — سيتم التخطي بأمان")
        return []


# ------------------------------------------------------------------------
# 5) بناء الهدف (Target) — التنبؤ بالاتجاه أو السعر بدون تسريب مستقبلي
# ------------------------------------------------------------------------
def build_target(data, horizon, target_type, min_move_threshold=0.0):
    """
    min_move_threshold: نسبة الحد الأدنى للحركة (مثلاً 0.003 = 0.3%) عشان نستبعد
    الحركات التافهة اللي هي نويز أكتر منها إشارة حقيقية. الصفوف اللي حركتها
    أصغر من العتبة تتشال بالكامل من التدريب والاختبار (Dead Zone).
    """
    data = data.copy()
    future_close = data["close"].shift(-horizon)
    pct_move = (future_close - data["close"]) / data["close"]

    if target_type == "direction":
        data["target"] = (future_close > data["close"]).astype(int)
        if min_move_threshold > 0:
            dead_zone_mask = pct_move.abs() < min_move_threshold
            data.loc[dead_zone_mask, "target"] = np.nan
    else:
        data["target"] = future_close

    # إزالة آخر horizon صفوف لأن الهدف فيها غير معروف فعلياً
    data = data.iloc[:-horizon].copy()
    return data


# ------------------------------------------------------------------------
# 6) تقسيم البيانات زمنياً (لا يوجد شافل عشوائي — احترام تسلسل الزمن)
# ------------------------------------------------------------------------
def time_split(data, split_ratio, test_start=None, test_end=None):
    data = data.dropna().reset_index(drop=True)

    if test_start or test_end:
        ts = pd.to_datetime(test_start, utc=True) if test_start else data["timestamp"].min()
        te = pd.to_datetime(test_end, utc=True) if test_end else data["timestamp"].max()
        test_mask = (data["timestamp"] >= ts) & (data["timestamp"] <= te)
        test_df = data[test_mask]
        train_df = data[data["timestamp"] < ts]
        print(f"[+] استخدام فترة اختبار محددة يدوياً: {ts.date()} -> {te.date()}")
    else:
        split_idx = int(len(data) * split_ratio)
        train_df = data.iloc[:split_idx]
        test_df = data.iloc[split_idx:]

    print(f"[+] بيانات التدريب: {len(train_df)} صف | بيانات الاختبار: {len(test_df)} صف")
    return train_df, test_df


# ------------------------------------------------------------------------
# 7) تدريب الموديلات
# ------------------------------------------------------------------------
FEATURE_BLACKLIST = ["timestamp", "target", "open", "high", "low", "close", "volume"]


def get_feature_columns(data):
    return [c for c in data.columns if c not in FEATURE_BLACKLIST]


def train_models(train_df, feature_cols, target_type, models_to_use):
    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    trained = {}

    # --- حساب نسبة توازن الفئات (لو مباشرة/صعود ونزول مش متساويين) ---
    pos_ratio = None
    if target_type == "direction":
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        pos_ratio = n_neg / max(n_pos, 1)
        print(f"[+] توازن الفئات: صعود={n_pos} | نزول={n_neg} | scale_pos_weight={pos_ratio:.3f}")

    if "lightgbm" in models_to_use and lgb is not None:
        print("[+] تدريب LightGBM...")
        params = dict(
            n_estimators=600, learning_rate=0.02, num_leaves=63,
            max_depth=-1, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbosity=-1,
        )
        if target_type == "direction":
            model = lgb.LGBMClassifier(**params, class_weight="balanced")
        else:
            model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        trained["lightgbm"] = model

    if "xgboost" in models_to_use and xgb is not None:
        print("[+] تدريب XGBoost...")
        params = dict(
            n_estimators=600, learning_rate=0.02, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, random_state=42,
        )
        if target_type == "direction":
            model = xgb.XGBClassifier(**params, eval_metric="logloss", scale_pos_weight=pos_ratio)
        else:
            model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        trained["xgboost"] = model

    return trained


# ------------------------------------------------------------------------
# 6.5) Walk-Forward Validation — تقييم أدق وأصعب من split واحد
# ------------------------------------------------------------------------
def walk_forward_validation(data, feature_cols, target_type, models_to_use, n_splits=5):
    """
    يقسم البيانات لـ n_splits فترات متتالية زمنياً (Expanding Window):
    كل فترة، يتدرب الموديل على كل البيانات اللي قبلها، ويتقيّم على الفترة الجاية.
    ده بيوريك هل أداء الموديل مستقر عبر الزمن، ولا كان مجرد حظ في split واحد.
    """
    data = data.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    n = len(data)
    fold_size = n // (n_splits + 1)

    if fold_size < 50:
        print("[!] بيانات غير كافية لعمل Walk-Forward بهذا العدد من الفترات")
        return []

    results = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = min(fold_size * (i + 1), n)
        train_fold = data.iloc[:train_end]
        test_fold = data.iloc[train_end:test_end]

        if len(test_fold) < 10:
            continue

        trained = train_models(train_fold, feature_cols, target_type, models_to_use)
        fold_result = {
            "fold": i,
            "train_start": train_fold["timestamp"].iloc[0],
            "train_end": train_fold["timestamp"].iloc[-1],
            "test_start": test_fold["timestamp"].iloc[0],
            "test_end": test_fold["timestamp"].iloc[-1],
            "n_test": len(test_fold),
        }

        for name, model in trained.items():
            X_test = test_fold[feature_cols]
            y_test = test_fold["target"]
            if target_type == "direction":
                pred = model.predict(X_test)
                fold_result[f"{name}_accuracy"] = accuracy_score(y_test, pred)
            else:
                pred = model.predict(X_test)
                fold_result[f"{name}_mae"] = mean_absolute_error(y_test, pred)

        results.append(fold_result)
        print(f"[+] Fold {i}/{n_splits} تم — نتائج: {fold_result}")

    return results


# ------------------------------------------------------------------------
# 8) التقييم + الـ Backtest الاستراتيجي
# ------------------------------------------------------------------------
def evaluate_models(trained_models, test_df, feature_cols, target_type):
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    predictions = {}
    print("\n" + "=" * 60)
    print("نتائج التقييم على بيانات الاختبار (لم يشوفها الموديل من قبل)")
    print("=" * 60)

    for name, model in trained_models.items():
        if target_type == "direction":
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            acc = accuracy_score(y_test, pred)
            prec = precision_score(y_test, pred, zero_division=0)
            rec = recall_score(y_test, pred, zero_division=0)
            f1 = f1_score(y_test, pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, proba)
            except Exception:
                auc = float("nan")
            print(f"\n[{name.upper()}]")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall   : {rec:.4f}")
            print(f"  F1-score : {f1:.4f}")
            print(f"  ROC-AUC  : {auc:.4f}")
            print(f"  (Baseline عشوائي متوقع تقريباً 0.50 accuracy)")
            predictions[name] = pred
        else:
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            mape = np.mean(np.abs((y_test - pred) / y_test)) * 100
            print(f"\n[{name.upper()}]")
            print(f"  MAE : {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            predictions[name] = pred

    # --- Ensemble بسيط (تصويت/متوسط) لو أكتر من موديل ---
    if len(trained_models) > 1:
        if target_type == "direction":
            stacked = np.mean([predictions[n] for n in predictions], axis=0)
            ensemble_pred = (stacked >= 0.5).astype(int)
            acc = accuracy_score(y_test, ensemble_pred)
            print(f"\n[ENSEMBLE (متوسط الموديلات)]")
            print(f"  Accuracy: {acc:.4f}")
            predictions["ensemble"] = ensemble_pred
        else:
            stacked = np.mean([predictions[n] for n in predictions], axis=0)
            mae = mean_absolute_error(y_test, stacked)
            print(f"\n[ENSEMBLE (متوسط الموديلات)]")
            print(f"  MAE: {mae:.4f}")
            predictions["ensemble"] = stacked

    return predictions


def backtest_strategy(test_df, predictions, target_type, output_dir):
    """محاكاة استراتيجية تداول بسيطة بناءً على تنبؤات الموديل، ومقارنتها بـ Buy & Hold"""
    if target_type != "direction":
        print("\n[!] الـ backtest الاستراتيجي متاح فقط عند TARGET_TYPE = 'direction'")
        return

    key = "ensemble" if "ensemble" in predictions else list(predictions.keys())[0]
    signal = predictions[key]

    df = test_df.reset_index(drop=True).copy()
    df["signal"] = signal
    df["market_return"] = df["close"].pct_change().shift(-1)
    df["strategy_return"] = df["market_return"] * df["signal"]

    df["cum_market"] = (1 + df["market_return"].fillna(0)).cumprod()
    df["cum_strategy"] = (1 + df["strategy_return"].fillna(0)).cumprod()

    final_market = df["cum_market"].iloc[-1]
    final_strategy = df["cum_strategy"].iloc[-1]

    print("\n" + "=" * 60)
    print("Backtest استراتيجي (على بيانات الاختبار فقط — بدون رسوم تداول)")
    print("=" * 60)
    print(f"  عائد Buy & Hold      : {(final_market - 1) * 100:.2f}%")
    print(f"  عائد استراتيجية الموديل: {(final_strategy - 1) * 100:.2f}%")
    print("  (تذكير: هذا بدون عمولات/سبريد، والواقع الفعلي غالباً أضعف)")

    # رسم بياني
    plt.figure(figsize=(11, 5))
    plt.plot(df.index, df["cum_market"], label="Buy & Hold")
    plt.plot(df.index, df["cum_strategy"], label="Model Strategy")
    plt.title("Backtest: Model Strategy vs Buy & Hold")
    plt.xlabel("Time steps (test period)")
    plt.ylabel("Cumulative return (x)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, "backtest_chart.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  تم حفظ رسم الـ backtest في: {out_path}")


def plot_feature_importance(trained_models, feature_cols, output_dir):
    for name, model in trained_models.items():
        if not hasattr(model, "feature_importances_"):
            continue
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=False).head(20)

        plt.figure(figsize=(9, 7))
        importances.sort_values().plot(kind="barh")
        plt.title(f"أهم 20 خاصية — {name}")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"feature_importance_{name}.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"[+] تم حفظ رسم أهمية الخصائص: {out_path}")


# ------------------------------------------------------------------------
# 8.5) تقرير HTML تفاعلي — تفتحه في أي متصفح على جهازك
# ------------------------------------------------------------------------
def generate_html_report(raw, featured, test_df, predictions, target_type, config, output_dir,
                          trained_models=None, feature_cols=None):
    print("[+] جاري بناء تقرير HTML تفاعلي...")

    plot_df = featured.copy()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        vertical_spacing=0.03,
        subplot_titles=("السعر + Bollinger Bands + EMA", "الحجم", "RSI", "MACD"),
    )

    # --- شارت الشموع ---
    fig.add_trace(go.Candlestick(
        x=plot_df["timestamp"], open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"], name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["bb_high"],
                              line=dict(color="rgba(150,150,255,0.5)", width=1),
                              name="BB High"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["bb_low"],
                              line=dict(color="rgba(150,150,255,0.5)", width=1),
                              name="BB Low", fill="tonexty",
                              fillcolor="rgba(150,150,255,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["ema_50"],
                              line=dict(color="orange", width=1.3), name="EMA 50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["ema_200"],
                              line=dict(color="purple", width=1.3), name="EMA 200"), row=1, col=1)

    # --- الحجم ---
    fig.add_trace(go.Bar(x=plot_df["timestamp"], y=plot_df["volume"],
                          name="Volume", marker_color="rgba(100,149,237,0.6)"), row=2, col=1)

    # --- RSI ---
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["rsi_14"],
                              line=dict(color="teal"), name="RSI 14"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    # --- MACD ---
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["macd"],
                              line=dict(color="blue"), name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["macd_signal"],
                              line=dict(color="orange"), name="Signal"), row=4, col=1)
    fig.add_trace(go.Bar(x=plot_df["timestamp"], y=plot_df["macd_diff"],
                          name="MACD Diff", marker_color="grey"), row=4, col=1)

    fig.update_layout(
        title=f"{config['SYMBOL']} — {config['TIMEFRAME']} — تقرير بيانات وتحليل فني",
        height=1000, template="plotly_dark", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05),
    )

    chart_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    # --- كارت التوصية الحية (شراء/بيع + نسبة التوقع) ---
    recommendation_html = "<p>لم يتم تدريب موديلات لعرض توصية حية.</p>"
    if target_type == "direction" and trained_models and feature_cols:
        last_row = featured.tail(1)[feature_cols]
        cards = ""
        probs = []
        for name, model in trained_models.items():
            p_up = float(model.predict_proba(last_row)[:, 1][0])
            probs.append(p_up)
            is_buy = p_up >= 0.5
            label = "شراء (BUY)" if is_buy else "بيع (SELL)"
            color = "#1b8f4d" if is_buy else "#c0392b"
            shown_pct = p_up * 100 if is_buy else (1 - p_up) * 100
            cards += f"""
            <div class="signal-card" style="border-color:{color};">
              <div class="signal-model">{name.upper()}</div>
              <div class="signal-label" style="color:{color};">{label}</div>
              <div class="signal-pct">نسبة التوقع: {shown_pct:.1f}%</div>
            </div>"""

        if len(trained_models) > 1:
            avg_p = sum(probs) / len(probs)
            is_buy = avg_p >= 0.5
            label = "شراء (BUY)" if is_buy else "بيع (SELL)"
            color = "#1b8f4d" if is_buy else "#c0392b"
            shown_pct = avg_p * 100 if is_buy else (1 - avg_p) * 100
            cards += f"""
            <div class="signal-card ensemble" style="border-color:{color};">
              <div class="signal-model">ENSEMBLE (الإجمالي)</div>
              <div class="signal-label" style="color:{color};">{label}</div>
              <div class="signal-pct">نسبة التوقع: {shown_pct:.1f}%</div>
            </div>"""

        last_ts = featured["timestamp"].iloc[-1]
        recommendation_html = f"""
        <div class="signals-row">{cards}</div>
        <p class="signal-note">التوصية مبنية على آخر بيانات متاحة حتى: {last_ts}</p>
        """

    # --- جدول ملخص أداء الموديلات (لو نتائج الاختبار جاهزة) ---
    summary_rows = ""
    if target_type == "direction" and predictions:
        y_test = test_df["target"]
        for name, pred in predictions.items():
            acc = accuracy_score(y_test, pred)
            summary_rows += f"<tr><td>{name}</td><td>{acc:.4f}</td></tr>"
    metrics_table = f"""
    <table class="metrics">
      <tr><th>الموديل</th><th>Accuracy</th></tr>
      {summary_rows}
    </table>
    """ if summary_rows else "<p>لا توجد نتائج تقييم متاحة (شغّل التدريب أولاً).</p>"

    last_row = featured.tail(200)
    data_table_rows = ""

    # بناء dict: timestamp -> نسبة صعود (لو الصف موجود في بيانات الاختبار)
    ts_to_prob = {}
    if target_type == "direction" and trained_models and feature_cols and len(test_df) > 0:
        best_model_name = "ensemble" if "ensemble" in predictions else list(trained_models.keys())[0]
        model_for_table = trained_models.get(best_model_name)
        if model_for_table is not None:
            probs_test = model_for_table.predict_proba(test_df[feature_cols])[:, 1]
            ts_to_prob = dict(zip(test_df["timestamp"], probs_test))

    for _, r in last_row.tail(30).iterrows():
        rec_cell = "-"
        if r["timestamp"] in ts_to_prob:
            p_up = ts_to_prob[r["timestamp"]]
            if p_up >= 0.5:
                rec_cell = f"<span style='color:#1b8f4d;'>شراء ({p_up*100:.1f}%)</span>"
            else:
                rec_cell = f"<span style='color:#c0392b;'>بيع ({(1-p_up)*100:.1f}%)</span>"
        data_table_rows += (
            f"<tr><td>{r['timestamp']}</td><td>{r['open']:.2f}</td>"
            f"<td>{r['high']:.2f}</td><td>{r['low']:.2f}</td>"
            f"<td>{r['close']:.2f}</td><td>{r['volume']:.2f}</td>"
            f"<td>{r.get('rsi_14', float('nan')):.2f}</td><td>{rec_cell}</td></tr>"
        )

    html_content = f"""
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<title>{config['SYMBOL']} — تقرير التحليل</title>
<style>
  body {{ background:#111; color:#eee; font-family: 'Segoe UI', Tahoma, sans-serif; margin:0; padding:20px; }}
  h1 {{ text-align:center; }}
  .metrics {{ margin: 20px auto; border-collapse: collapse; width: 300px; }}
  .metrics th, .metrics td {{ border:1px solid #444; padding:8px 12px; text-align:center; }}
  table.data {{ width:100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }}
  table.data th, table.data td {{ border:1px solid #333; padding:5px 8px; text-align:center; }}
  table.data th {{ background:#222; }}
  .section {{ max-width: 1100px; margin: 20px auto; }}
  .warning {{ background:#332200; border:1px solid #886600; padding:12px; border-radius:8px; margin:20px auto; max-width:1100px; }}
  .signals-row {{ display:flex; gap:16px; flex-wrap:wrap; justify-content:center; }}
  .signal-card {{ background:#1a1a1a; border:2px solid #555; border-radius:10px; padding:16px 24px; text-align:center; min-width:180px; }}
  .signal-card.ensemble {{ border-width:3px; }}
  .signal-model {{ font-size:13px; color:#aaa; margin-bottom:6px; }}
  .signal-label {{ font-size:22px; font-weight:bold; margin-bottom:4px; }}
  .signal-pct {{ font-size:14px; color:#ccc; }}
  .signal-note {{ text-align:center; color:#888; font-size:12px; margin-top:10px; }}
</style>
</head>
<body>
<h1>{config['SYMBOL']} — تقرير بيانات وتحليل ({config['TIMEFRAME']})</h1>
<div class="section">
  <h2>التوصية الحالية (بناءً على آخر شمعة)</h2>
  {recommendation_html}
</div>
<div class="section">{chart_html}</div>
<div class="section">
  <h2>ملخص أداء الموديلات (على بيانات الاختبار)</h2>
  {metrics_table}
</div>
<div class="section">
  <h2>آخر 30 صف من البيانات الخام + توصية كل صف (على فترة الاختبار)</h2>
  <table class="data">
    <tr><th>الوقت</th><th>Open</th><th>High</th><th>Low</th><th>Close</th><th>Volume</th><th>RSI 14</th><th>التوصية</th></tr>
    {data_table_rows}
  </table>
</div>
<div class="warning">
  تذكير: هذا تقرير تحليلي وتعليمي، وليس نصيحة استثمارية. الأداء في السوق الحقيقي قد يختلف تماماً عن الباك تيست.
</div>
</body>
</html>
"""
    out_path = os.path.join(output_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] تم حفظ تقرير الويب التفاعلي في: {out_path}")
    print("    افتحه بالمتصفح مباشرة (دبل كليك عليه) لتشوف الشارت والبيانات.")
    return out_path


# ------------------------------------------------------------------------
# 9) التشغيل الكامل
# ------------------------------------------------------------------------
def main():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    print("\n" + "#" * 60)
    print("# Crypto ML Predictor — تشغيل السكربت")
    print("#" * 60)
    print(f"العملة: {CONFIG['SYMBOL']} | الفريم: {CONFIG['TIMEFRAME']} | سنوات البيانات: {CONFIG['YEARS_BACK']}")
    if CONFIG["TEST_START_DATE"] or CONFIG["TEST_END_DATE"]:
        print(f"فترة الاختبار المحددة يدوياً: {CONFIG['TEST_START_DATE']} -> {CONFIG['TEST_END_DATE']}")
    print("#" * 60 + "\n")

    raw = fetch_binance_ohlcv(CONFIG["SYMBOL"], CONFIG["TIMEFRAME"], CONFIG["YEARS_BACK"])
    if len(raw) < 300:
        print("[!] بيانات قليلة جداً — تأكد من اسم الرمز والفريم الزمني")
        return

    featured = build_features(raw)

    if CONFIG["USE_HIGHER_TIMEFRAME_TREND"]:
        featured = add_higher_timeframe_trend(
            featured, CONFIG["SYMBOL"], CONFIG["TIMEFRAME"], CONFIG["YEARS_BACK"], CONFIG["HIGHER_TIMEFRAME"]
        )

    if CONFIG["USE_CROSS_ASSET_FEATURES"]:
        featured = add_cross_asset_features(
            featured, CONFIG["TIMEFRAME"], CONFIG["YEARS_BACK"], CONFIG["CROSS_ASSETS"]
        )
        featured = add_rolling_correlation_with_btc(featured, window=30)

    featured = add_tron_onchain_features(featured, CONFIG["SYMBOL"])
    featured = add_swing_points(featured, order=5)
    featured = add_candlestick_patterns(featured)
    featured = add_liquidity_features(featured, window=20)

    targeted = build_target(
        featured, CONFIG["PREDICTION_HORIZON"], CONFIG["TARGET_TYPE"], CONFIG["MIN_MOVE_THRESHOLD"]
    )

    feature_cols = get_feature_columns(targeted)

    # --- Walk-Forward Validation: تقييم أدق قبل التدريب النهائي ---
    if CONFIG["USE_WALK_FORWARD"] and CONFIG["TARGET_TYPE"] == "direction":
        print("\n" + "=" * 60)
        print("Walk-Forward Validation — تقييم الاستقرار عبر فترات متعددة")
        print("=" * 60)
        wf_results = walk_forward_validation(
            targeted, feature_cols, CONFIG["TARGET_TYPE"], CONFIG["MODELS_TO_USE"], CONFIG["WALK_FORWARD_SPLITS"]
        )
        if wf_results:
            for name in CONFIG["MODELS_TO_USE"]:
                key = f"{name}_accuracy"
                accs = [r[key] for r in wf_results if key in r]
                if accs:
                    print(f"  [{name}] متوسط Accuracy عبر {len(accs)} فترات: {np.mean(accs):.4f} "
                          f"(± {np.std(accs):.4f})")

    train_df, test_df = time_split(
        targeted,
        CONFIG["TRAIN_TEST_SPLIT_RATIO"],
        CONFIG["TEST_START_DATE"],
        CONFIG["TEST_END_DATE"],
    )

    if len(train_df) < 100 or len(test_df) < 20:
        print("[!] بيانات التدريب/الاختبار غير كافية — راجع الفترة الزمنية المحددة")
        return

    trained_models = train_models(train_df, feature_cols, CONFIG["TARGET_TYPE"], CONFIG["MODELS_TO_USE"])

    if not trained_models:
        print("[!] لم يتم تدريب أي موديل — تأكد من تثبيت lightgbm/xgboost")
        return

    predictions = evaluate_models(trained_models, test_df, feature_cols, CONFIG["TARGET_TYPE"])
    backtest_strategy(test_df, predictions, CONFIG["TARGET_TYPE"], CONFIG["OUTPUT_DIR"])
    plot_feature_importance(trained_models, feature_cols, CONFIG["OUTPUT_DIR"])
    generate_html_report(raw, featured, test_df, predictions, CONFIG["TARGET_TYPE"], CONFIG, CONFIG["OUTPUT_DIR"],
                          trained_models=trained_models, feature_cols=feature_cols)

    # حفظ آخر تنبؤ "حي" (للشمعة القادمة بعد آخر بيانات متاحة)
    last_row = targeted.iloc[[-1]][feature_cols]
    print("\n" + "=" * 60)
    print("تنبؤ الشمعة القادمة (Live-ish، بناءً على آخر بيانات متاحة)")
    print("=" * 60)
    for name, model in trained_models.items():
        if CONFIG["TARGET_TYPE"] == "direction":
            p = model.predict_proba(last_row)[:, 1][0]
            direction = "صعود ⬆️" if p >= 0.5 else "نزول ⬇️"
            print(f"  [{name}] احتمال الصعود: {p:.3f} -> {direction}")
        else:
            p = model.predict(last_row)[0]
            print(f"  [{name}] السعر المتوقع: {p:.2f}")

    print("\n[!] تذكير أخير: هذه تنبؤات إحصائية وليست ضمانات. لا تتخذ قرارات مالية بناءً عليها فقط.")


if __name__ == "__main__":
    main()
