"""
========================================================================
 TRX Pro Predictor — كل الأدوات المتقدمة، على فريم واحد تختاره
========================================================================
نفس قوة النسخة الشاملة (Ensemble + Walk-Forward + قمم/قيعان + سيولة +
BTC/ETH + مقارنة Dead Zone)، لكن منظّمة عشان تشتغل على فريم واحد بس
في كل مرة (1 ساعة أو 8 ساعات) بدل تشغيل الاتنين سوا — ده بيقلل الحمل
للنص تقريباً ويخلي السكربت يقدر يخلّص فعلياً.
========================================================================
"""

import os
import sys
from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_ml_predictor as core

import streamlit as st

SYMBOL = "TRX/USDT"
CAIRO_TZ = ZoneInfo("Africa/Cairo")


def to_cairo(ts_utc):
    ts = pd.Timestamp(ts_utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(CAIRO_TZ)


def timeframe_duration(tf):
    return timedelta(minutes={"1h": 60, "8h": 480}.get(tf, 60))


def quick_horizon_prediction(featured, dead_zone, horizon, model_seed_features=None):
    """توقع سريع وخفيف (موديل واحد بس) لشمعة على مدى h — مستخدم للشمعتين 2 و3 في الجدول"""
    targeted = core.build_target(featured, horizon, "direction", dead_zone)
    feature_cols = model_seed_features if model_seed_features else core.get_feature_columns(targeted)
    feature_cols = [c for c in feature_cols if c in targeted.columns]
    train_df, test_df = core.time_split(targeted, 0.85, None, None)

    if len(train_df) < 100 or len(test_df) < 20:
        return None

    model = core.lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=-1, class_weight="balanced",
    )
    model.fit(train_df[feature_cols], train_df["target"])

    X_test, y_test = test_df[feature_cols], test_df["target"].values
    test_probs = model.predict_proba(X_test)[:, 1]
    correct = ((test_probs >= 0.5).astype(int) == y_test).astype(int)
    last_n = min(100, len(correct))
    acc = correct[-last_n:].mean() * 100 if last_n > 0 else None

    last_row = featured.tail(1)[feature_cols]
    p_up = float(model.predict_proba(last_row)[:, 1][0])
    side = "شراء ⬆️" if p_up >= 0.5 else "بيع ⬇️"
    conf = p_up * 100 if p_up >= 0.5 else (1 - p_up) * 100
    return {"side": side, "conf": conf, "acc": acc}



st.set_page_config(page_title="TRX Pro Predictor", layout="wide", page_icon="🎯")

st.title("🎯 TRX Pro Predictor")
st.caption("كل الأدوات المتقدمة، مركّزة على فريم واحد تختاره — أسرع وأكفى من تشغيل فريمين مع بعض")

# ------------------------------------------------------------------------
# الشريط الجانبي
# ------------------------------------------------------------------------
st.sidebar.title("⚙️ الإعدادات")
timeframe_choice = st.sidebar.radio("اختار الفريم الزمني", ["1 ساعة", "8 ساعات"], index=1)
timeframe = "1h" if timeframe_choice == "1 ساعة" else "8h"
default_dz = 0.2 if timeframe == "1h" else 0.4

years_back = st.sidebar.slider("عدد سنين البيانات", 0.5, 3.0, 1.5, step=0.5)
dead_zone_pct = st.sidebar.number_input(
    "Dead Zone % (استخدم زرار المقارنة تحت لو مش متأكد من أفضل قيمة)",
    min_value=0.0, max_value=3.0, value=default_dz, step=0.05,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 الأدوات المتقدمة")
use_advanced_features = st.sidebar.checkbox(
    "تفعيل BTC/ETH + اتجاه يومي + قمم/قيعان + سيولة", value=True,
)
use_walk_forward = st.sidebar.checkbox("تفعيل Walk-Forward Validation", value=True)
wf_splits = st.sidebar.slider("عدد فترات Walk-Forward", 3, 6, 3) if use_walk_forward else 0

run_btn = st.sidebar.button("🎯 تشغيل التحليل الكامل", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 مقارنة Dead Zone (اختياري)")
dz_max = st.sidebar.slider("أقصى قيمة للمقارنة %", 0.5, 3.0, 1.5, step=0.1)
dz_step = st.sidebar.selectbox("الخطوة %", [0.1, 0.2, 0.25], index=0)
compare_btn = st.sidebar.button("🔍 قارن كل قيم Dead Zone", use_container_width=True)


# ------------------------------------------------------------------------
# فنكشن مشتركة: تجهيز البيانات مرة واحدة (يُعاد استخدامها للتحليل والمقارنة)
# ------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_and_featurize(timeframe, years_back, use_advanced):
    raw = core.fetch_binance_ohlcv(SYMBOL, timeframe, years_back)
    if len(raw) < 300:
        return None
    featured = core.build_features(raw)
    if use_advanced:
        featured = core.add_higher_timeframe_trend(featured, SYMBOL, timeframe, years_back, "1d")
        featured = core.add_cross_asset_features(featured, timeframe, years_back, ["BTC/USDT", "ETH/USDT"])
        featured = core.add_rolling_correlation_with_btc(featured, window=30)
        featured = core.add_swing_points(featured, order=5)
        featured = core.add_candlestick_patterns(featured)
        featured = core.add_liquidity_features(featured, window=20)
    return featured


# ------------------------------------------------------------------------
# التحليل الكامل (إعداد واحد بس)
# ------------------------------------------------------------------------
def run_full_analysis(featured, dead_zone_pct, use_wf, wf_splits, progress_cb=None):
    dead_zone = dead_zone_pct / 100.0
    targeted = core.build_target(featured, 1, "direction", dead_zone)
    feature_cols = core.get_feature_columns(targeted)

    wf_results = None
    if use_wf:
        if progress_cb:
            progress_cb("تشغيل Walk-Forward Validation...")
        wf_results = core.walk_forward_validation(targeted, feature_cols, "direction", ["lightgbm", "xgboost"], wf_splits)

    train_df, test_df = core.time_split(targeted, 0.85, None, None)
    if len(train_df) < 100 or len(test_df) < 30:
        return {"error": "بيانات تدريب/اختبار غير كافية بعد التصفية — قلّل Dead Zone أو زوّد عدد السنين"}

    val_frac = 0.15
    split_idx = int(len(train_df) * (1 - val_frac))
    sub_train, val = train_df.iloc[:split_idx], train_df.iloc[split_idx:]

    if progress_cb:
        progress_cb("اختيار أقوى الخصائص...")
    probe = core.train_models(sub_train, feature_cols, "direction", ["lightgbm"])
    importances = pd.Series(probe["lightgbm"].feature_importances_, index=feature_cols)
    keep_n = max(15, int(len(feature_cols) * 0.6))
    selected_features = importances.sort_values(ascending=False).head(keep_n).index.tolist()

    if progress_cb:
        progress_cb("ضبط عتبة القرار المثلى...")
    trained_val = core.train_models(sub_train, selected_features, "direction", ["lightgbm", "xgboost"])
    val_probs_by_model = {name: m.predict_proba(val[selected_features])[:, 1] for name, m in trained_val.items()}

    model_weights = {"lightgbm": 0.5, "xgboost": 0.5}
    if wf_results:
        wf_accs = {}
        for name in ["lightgbm", "xgboost"]:
            accs = [wf[f"{name}_accuracy"] for wf in wf_results if f"{name}_accuracy" in wf]
            if accs:
                wf_accs[name] = np.mean(accs)
        if len(wf_accs) == 2 and sum(wf_accs.values()) > 0:
            total = sum(wf_accs.values())
            model_weights = {name: acc / total for name, acc in wf_accs.items()}

    val_ensemble_probs = sum(val_probs_by_model[n] * model_weights[n] for n in val_probs_by_model)
    y_val = val["target"].values
    best_thr, best_acc = 0.5, 0.0
    for thr in np.arange(0.35, 0.66, 0.01):
        acc = ((val_ensemble_probs >= thr).astype(int) == y_val).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    if progress_cb:
        progress_cb("تدريب الموديلات النهائية...")
    trained = core.train_models(train_df, selected_features, "direction", ["lightgbm", "xgboost"])
    if not trained:
        return {"error": "فشل تدريب الموديلات"}

    X_test = test_df[selected_features]
    y_test = test_df["target"].values
    probs_by_model = {name: m.predict_proba(X_test)[:, 1] for name, m in trained.items()}
    ensemble_probs = sum(probs_by_model[n] * model_weights[n] for n in probs_by_model)
    ensemble_preds = (ensemble_probs >= best_thr).astype(int)
    correct = (ensemble_preds == y_test).astype(int)

    confidence_margin = 0.07
    is_confident = np.abs(ensemble_probs - best_thr) >= confidence_margin
    last_n = min(100, len(correct))
    acc_last_n = correct[-last_n:].mean() * 100
    confident_slice = is_confident[-last_n:]
    if confident_slice.sum() >= 10:
        acc_confident_only = correct[-last_n:][confident_slice].mean() * 100
        n_confident = int(confident_slice.sum())
    else:
        acc_confident_only, n_confident = None, 0

    last_row = featured.tail(1)[selected_features]
    live_probs = {name: float(m.predict_proba(last_row)[:, 1][0]) for name, m in trained.items()}
    live_ensemble_p = sum(live_probs[n] * model_weights[n] for n in live_probs)
    live_is_confident = abs(live_ensemble_p - best_thr) >= confidence_margin
    live_side = "شراء" if live_ensemble_p >= best_thr else "بيع"
    live_conf = live_ensemble_p * 100 if live_side == "شراء" else (1 - live_ensemble_p) * 100

    last_r = featured.iloc[-1]

    return {
        "error": None,
        "acc_last_n": acc_last_n, "n_used": last_n, "n_test": len(test_df),
        "acc_confident_only": acc_confident_only, "n_confident": n_confident,
        "live_side": live_side, "live_conf": live_conf, "live_is_confident": live_is_confident,
        "live_probs": live_probs, "model_weights": model_weights, "best_thr": best_thr,
        "n_features_selected": len(selected_features), "n_features_total": len(feature_cols),
        "wf_results": wf_results,
        "dist_swing_high": last_r.get("dist_to_last_swing_high"),
        "dist_swing_low": last_r.get("dist_to_last_swing_low"),
        "volume_zscore": last_r.get("volume_zscore"),
        "corr_btc": last_r.get("corr_with_btc_30"),
        "last_ts": featured["timestamp"].iloc[-1],
    }


# ------------------------------------------------------------------------
# تشغيل: التحليل الكامل
# ------------------------------------------------------------------------
if run_btn:
    status = st.empty()
    with st.spinner("جاري تحميل البيانات وبناء المؤشرات..."):
        try:
            featured = load_and_featurize(timeframe, years_back, use_advanced_features)
        except Exception as e:
            st.error(f"❌ فشل تحميل البيانات: {e}")
            st.stop()

    if featured is None:
        st.error("بيانات قليلة جداً. جرّب مدة أطول.")
        st.stop()

    def cb(msg):
        status.info(f"⏳ {msg}")

    result = run_full_analysis(featured, dead_zone_pct, use_walk_forward, wf_splits, cb)

    # --- توقع سريع للشمعتين 2 و3 (بالإضافة للشمعة 1 اللي التحليل الكامل غطاها) ---
    cb("توقع الشمعة رقم 2...")
    h2 = quick_horizon_prediction(featured, dead_zone_pct / 100.0, 2)
    cb("توقع الشمعة رقم 3...")
    h3 = quick_horizon_prediction(featured, dead_zone_pct / 100.0, 3)
    status.empty()

    st.session_state["result"] = result
    st.session_state["multistep"] = {1: result, 2: h2, 3: h3}
    st.session_state["last_ts"] = featured["timestamp"].iloc[-1]
    st.session_state["timeframe"] = timeframe
    st.session_state["timeframe_choice"] = timeframe_choice
    st.session_state["dead_zone_pct"] = dead_zone_pct

# ------------------------------------------------------------------------
# تشغيل: مقارنة Dead Zone (خفيفة — موديل واحد بس لكل قيمة)
# ------------------------------------------------------------------------
if compare_btn:
    with st.spinner("جاري تحميل البيانات..."):
        try:
            featured_cmp = load_and_featurize(timeframe, years_back, use_advanced_features)
        except Exception as e:
            st.error(f"❌ فشل تحميل البيانات: {e}")
            st.stop()

    if featured_cmp is None:
        st.error("بيانات قليلة جداً.")
        st.stop()

    candidate_values = [round(v, 2) for v in np.arange(0.0, dz_max + dz_step / 2, dz_step)]
    cmp_results = []
    progress = st.progress(0, text="بدء المقارنة...")

    for i, dz_pct in enumerate(candidate_values):
        progress.progress(i / len(candidate_values), text=f"تجربة Dead Zone = {dz_pct}% ({i+1}/{len(candidate_values)})")
        dz = dz_pct / 100.0
        targeted_cmp = core.build_target(featured_cmp, 1, "direction", dz)
        feature_cols_cmp = core.get_feature_columns(targeted_cmp)
        train_df_cmp, test_df_cmp = core.time_split(targeted_cmp, 0.85, None, None)

        if len(train_df_cmp) < 100 or len(test_df_cmp) < 30:
            cmp_results.append({"dz": dz_pct, "acc": None})
            continue

        model_cmp = core.lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbosity=-1, class_weight="balanced",
        )
        model_cmp.fit(train_df_cmp[feature_cols_cmp], train_df_cmp["target"])
        probs_cmp = model_cmp.predict_proba(test_df_cmp[feature_cols_cmp])[:, 1]
        preds_cmp = (probs_cmp >= 0.5).astype(int)
        correct_cmp = (preds_cmp == test_df_cmp["target"].values).astype(int)
        last_n_cmp = min(100, len(correct_cmp))
        cmp_results.append({"dz": dz_pct, "acc": correct_cmp[-last_n_cmp:].mean() * 100, "n_used": last_n_cmp})

    progress.empty()
    st.session_state["cmp_results"] = cmp_results
    st.session_state["cmp_timeframe_choice"] = timeframe_choice

# ------------------------------------------------------------------------
# عرض نتائج المقارنة (لو موجودة)
# ------------------------------------------------------------------------
if "cmp_results" in st.session_state:
    st.header(f"🔍 مقارنة Dead Zone — {st.session_state['cmp_timeframe_choice']}")
    valid = [r for r in st.session_state["cmp_results"] if r["acc"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["acc"])
        st.success(f"🏆 أفضل قيمة: **Dead Zone = {best['dz']}%** بدقة **{best['acc']:.1f}%**")
        cmp_table = [{"Dead Zone %": r["dz"], "الدقة": f"{r['acc']:.1f}%" if r["acc"] is not None else "غير كافٍ"}
                     for r in sorted(st.session_state["cmp_results"], key=lambda x: (x["acc"] is None, -(x["acc"] or 0)))]
        st.dataframe(pd.DataFrame(cmp_table), use_container_width=True, hide_index=True)
        st.info(f"💡 استخدم القيمة دي في خانة «Dead Zone %» فوق، وبعدين دوس «تشغيل التحليل الكامل».")
    else:
        st.error("مفيش قيمة عطت بيانات كافية.")
    st.markdown("---")

# ------------------------------------------------------------------------
# عرض نتائج التحليل الكامل (لو موجودة)
# ------------------------------------------------------------------------
if "result" in st.session_state:
    r = st.session_state["result"]
    st.header(f"🎯 التحليل الكامل — {st.session_state['timeframe_choice']} (Dead Zone {st.session_state['dead_zone_pct']}%)")

    last_ts_cairo = to_cairo(st.session_state["last_ts"])
    st.success(f"✅ آخر بيانات فعلية حتى: **{last_ts_cairo.strftime('%Y-%m-%d %H:%M')}** (توقيت القاهرة)")

    # --- جدول توقع أقرب 3 شمعات (بتوقيت القاهرة) ---
    st.markdown("### 📋 جدول توقع أقرب 3 شمعات")
    duration = timeframe_duration(st.session_state["timeframe"])
    multistep = st.session_state.get("multistep", {})
    table_rows = []
    for h in [1, 2, 3]:
        candle_start = last_ts_cairo + duration * (h - 1)
        candle_end = candle_start + duration
        data = multistep.get(h)
        if h == 1 and data and not data.get("error"):
            side, conf, acc = data["live_side"] + (" ⬆️" if data["live_side"] == "شراء" else " ⬇️"), data["live_conf"], data["acc_last_n"]
        elif h != 1 and data:
            side, conf, acc = data["side"], data["conf"], data["acc"]
        else:
            side, conf, acc = "غير متاح", None, None
        table_rows.append({
            "الشمعة #": h,
            "من الساعة": candle_start.strftime("%Y-%m-%d %H:%M"),
            "إلى الساعة": candle_end.strftime("%H:%M"),
            "التوقع": side,
            "نسبة الثقة": f"{conf:.1f}%" if conf is not None else "-",
            "دقة تاريخية": f"{acc:.1f}%" if acc is not None else "-",
        })
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.caption(
        "كل الأوقات بتوقيت القاهرة المحلي. الشمعة #1 من التحليل الكامل (Ensemble قوي)، "
        "الشمعتين #2 و#3 من توقع سريع (موديل خفيف) لإعطاء نظرة أبعد."
    )
    st.markdown("---")

    if r.get("error"):
        st.error(f"❌ {r['error']}")
    else:
        color = "🟢" if r["live_side"] == "شراء" else "🔴"
        st.markdown(f"### {color} التوصية الحالية: **{r['live_side']}** — ثقة {r['live_conf']:.1f}%")
        if not r["live_is_confident"]:
            st.warning("⚪ **منطقة حياد** — الموديل مش واثق بشكل كافٍ حالياً.")
        cols = st.columns(len(r["live_probs"]))
        for i, (name, p) in enumerate(r["live_probs"].items()):
            side = "شراء" if p >= r["best_thr"] else "بيع"
            conf = p * 100 if side == "شراء" else (1 - p) * 100
            weight = r["model_weights"].get(name, 0.5)
            cols[i].metric(f"{name.upper()} (وزن {weight*100:.0f}%)", side, f"{conf:.1f}% ثقة")
        st.caption(f"عتبة القرار: {r['best_thr']:.2f} | "
                   f"خصائص مستخدمة: {r['n_features_selected']}/{r['n_features_total']}")

        st.markdown("---")
        st.markdown(f"### 📊 دقة آخر {r['n_used']} توقع (Ensemble): **{r['acc_last_n']:.1f}%**")
        if r["acc_confident_only"] is not None:
            st.markdown(f"**🎯 دقة التوصيات الواثقة فقط:** **{r['acc_confident_only']:.1f}%** "
                        f"(على {r['n_confident']} من أصل {r['n_used']})")

        if r.get("wf_results"):
            st.markdown("### 🔁 استقرار الأداء (Walk-Forward)")
            wf_table = []
            for wf in r["wf_results"]:
                row = {"الفترة #": wf["fold"], "من": to_cairo(wf["test_start"]).strftime("%Y-%m-%d %H:%M"),
                       "إلى": to_cairo(wf["test_end"]).strftime("%Y-%m-%d %H:%M")}
                for name in ["lightgbm", "xgboost"]:
                    if f"{name}_accuracy" in wf:
                        row[f"Acc ({name})"] = round(wf[f"{name}_accuracy"], 3)
                wf_table.append(row)
            st.dataframe(pd.DataFrame(wf_table), use_container_width=True, hide_index=True)

        st.markdown("### 🧭 سياق إضافي")
        if pd.notna(r.get("dist_swing_high")):
            st.write(f"📍 السعر أبعد من آخر **قمة** بـ **{r['dist_swing_high']*100:.2f}%**")
        if pd.notna(r.get("dist_swing_low")):
            st.write(f"📍 السعر أبعد من آخر **قاعدة** بـ **{r['dist_swing_low']*100:.2f}%**")
        if pd.notna(r.get("volume_zscore")):
            vz = r["volume_zscore"]
            st.write("💧 **سيولة قوية دخلت السوق**" if vz > 2 else ("💧 **سيولة ضعيفة**" if vz < -1.5 else "💧 سيولة طبيعية"))
        if pd.notna(r.get("corr_btc")):
            st.write(f"📎 ارتباط بـ BTC: **{r['corr_btc']:.2f}**")

    st.markdown("---")
    st.warning("⚠️ توقعات إحصائية تعليمية وليست نصيحة استثمارية.")

if "result" not in st.session_state and "cmp_results" not in st.session_state:
    st.info("👈 اضبط الإعدادات، وبعدين دوس «تشغيل التحليل الكامل» أو «قارن كل قيم Dead Zone».")
