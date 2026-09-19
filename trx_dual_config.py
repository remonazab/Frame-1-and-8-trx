"""
========================================================================
 TRX Dual-Config Predictor — تحليل مركّز على إعدادين مختارين بعناية
========================================================================
بدل تجربة كل الاحتمالات، السكربت ده مخصص بس على الإعدادين اللي أظهروا
أقوى نتائج في تجاربك:
  1) فريم 8 ساعات + Dead Zone 0.4%
  2) فريم 1 ساعة  + Dead Zone 0.2%

وبيستخدم كل أدوات التحسين المتاحة على الإعدادين دول تحديداً:
- Ensemble من LightGBM + XGBoost معاً
- Walk-Forward Validation (تأكيد الاستقرار عبر فترات متعددة)
- قمم وقيعان (Swing Points) + أنماط شموع + مراقبة سيولة
- اتجاه فريم زمني أعلى (يومي) + ربط بحركة BTC/ETH
========================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crypto_ml_predictor as core

import streamlit as st
from sklearn.metrics import accuracy_score

SYMBOL = "TRX/USDT"

st.set_page_config(page_title="TRX Dual-Config Predictor", layout="wide", page_icon="🎯")

# ------------------------------------------------------------------------
# الإعدادان الثابتان (مبنيّين على ملاحظاتك الفعلية)
# ------------------------------------------------------------------------
CONFIGS = [
    {"key": "8h", "label": "8 ساعات", "timeframe": "8h", "dead_zone_pct": 0.4},
    {"key": "1h", "label": "1 ساعة", "timeframe": "1h", "dead_zone_pct": 0.2},
]

st.title("🎯 TRX Dual-Config Predictor")
st.caption("تحليل مركّز على إعدادين محددين بعناية، بكل أدوات التحسين المتاحة")

# ------------------------------------------------------------------------
# الشريط الجانبي
# ------------------------------------------------------------------------
st.sidebar.title("⚙️ إعدادات عامة")
years_back = st.sidebar.slider("عدد سنين البيانات", 0.5, 3.0, 1.5, step=0.5)
use_walk_forward = st.sidebar.checkbox("تفعيل Walk-Forward Validation (تأكيد الاستقرار)", value=True)
wf_splits = st.sidebar.slider("عدد فترات Walk-Forward", 3, 6, 4) if use_walk_forward else 0
run_btn = st.sidebar.button("🔄 تحديث وتشغيل التحليل الكامل", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "الإعدادان الثابتان:\n"
    "- 8 ساعات → Dead Zone 0.4%\n"
    "- 1 ساعة → Dead Zone 0.2%\n\n"
    "بُنيا على أفضل نتائج لاحظتها فعلياً، وكل الأدوات (Ensemble, Walk-Forward, "
    "قمم/قيعان, سيولة, BTC/ETH) مفعّلة تلقائياً على الاثنين."
)


# ------------------------------------------------------------------------
# فنكشن تحليل كامل لإعداد واحد
# ------------------------------------------------------------------------
def analyze_config(cfg, years_back, use_wf, wf_splits, progress_cb=None):
    timeframe = cfg["timeframe"]
    dead_zone = cfg["dead_zone_pct"] / 100.0

    if progress_cb:
        progress_cb(f"[{cfg['label']}] تحميل بيانات TRX...")
    raw = core.fetch_binance_ohlcv(SYMBOL, timeframe, years_back)
    if len(raw) < 300:
        return {"error": "بيانات غير كافية"}

    if progress_cb:
        progress_cb(f"[{cfg['label']}] بناء المؤشرات الفنية...")
    featured = core.build_features(raw)

    if progress_cb:
        progress_cb(f"[{cfg['label']}] إضافة اتجاه الفريم الأعلى...")
    featured = core.add_higher_timeframe_trend(featured, SYMBOL, timeframe, years_back, "1d")

    if progress_cb:
        progress_cb(f"[{cfg['label']}] إضافة ميزات BTC/ETH...")
    featured = core.add_cross_asset_features(featured, timeframe, years_back, ["BTC/USDT", "ETH/USDT"])
    featured = core.add_rolling_correlation_with_btc(featured, window=30)

    if progress_cb:
        progress_cb(f"[{cfg['label']}] كشف القمم والقيعان والأنماط والسيولة...")
    featured = core.add_swing_points(featured, order=5)
    featured = core.add_candlestick_patterns(featured)
    featured = core.add_liquidity_features(featured, window=20)

    targeted = core.build_target(featured, 1, "direction", dead_zone)
    feature_cols = core.get_feature_columns(targeted)

    wf_results = None
    if use_wf:
        if progress_cb:
            progress_cb(f"[{cfg['label']}] تشغيل Walk-Forward Validation...")
        wf_results = core.walk_forward_validation(
            targeted, feature_cols, "direction", ["lightgbm", "xgboost"], wf_splits
        )

    train_df, test_df = core.time_split(targeted, 0.85, None, None)
    if len(train_df) < 100 or len(test_df) < 30:
        return {"error": "بيانات تدريب/اختبار غير كافية بعد التصفية"}

    # --- تقسيم إضافي داخل بيانات التدريب فقط (Validation) لضبط العتبة واختيار الخصائص ---
    # ده منفصل تماماً عن test_df عشان مفيش أي تسريب بيانات للنتيجة النهائية
    val_frac = 0.15
    split_idx = int(len(train_df) * (1 - val_frac))
    sub_train, val = train_df.iloc[:split_idx], train_df.iloc[split_idx:]

    if progress_cb:
        progress_cb(f"[{cfg['label']}] اختيار أقوى الخصائص...")
    trained_probe = core.train_models(sub_train, feature_cols, "direction", ["lightgbm"])
    importances = pd.Series(trained_probe["lightgbm"].feature_importances_, index=feature_cols)
    keep_n = max(15, int(len(feature_cols) * 0.6))
    selected_features = importances.sort_values(ascending=False).head(keep_n).index.tolist()

    if progress_cb:
        progress_cb(f"[{cfg['label']}] ضبط عتبة القرار المثلى...")
    trained_val = core.train_models(sub_train, selected_features, "direction", ["lightgbm", "xgboost"])
    val_probs_by_model = {name: m.predict_proba(val[selected_features])[:, 1] for name, m in trained_val.items()}

    # وزن كل موديل حسب أدائه في Walk-Forward (لو متاح)، وإلا وزن متساوي
    model_weights = {"lightgbm": 0.5, "xgboost": 0.5}
    if wf_results:
        wf_accs = {}
        for name in ["lightgbm", "xgboost"]:
            key = f"{name}_accuracy"
            accs = [wf[key] for wf in wf_results if key in wf]
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
        progress_cb(f"[{cfg['label']}] تدريب الموديلات النهائية (LightGBM + XGBoost)...")
    trained = core.train_models(train_df, selected_features, "direction", ["lightgbm", "xgboost"])
    if not trained:
        return {"error": "فشل تدريب الموديلات"}

    X_test = test_df[selected_features]
    y_test = test_df["target"].values
    probs_by_model = {name: m.predict_proba(X_test)[:, 1] for name, m in trained.items()}
    ensemble_probs = sum(probs_by_model[n] * model_weights[n] for n in probs_by_model)
    ensemble_preds = (ensemble_probs >= best_thr).astype(int)
    correct = (ensemble_preds == y_test).astype(int)

    # منطقة الحياد: لو الاحتمال قريب جداً من العتبة، نعتبره "مش واثق" ونستبعده من حساب دقة الثقة العالية
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

    # التوصية الحية (آخر شمعة)
    last_row = featured.tail(1)[selected_features]
    live_probs = {name: float(m.predict_proba(last_row)[:, 1][0]) for name, m in trained.items()}
    live_ensemble_p = sum(live_probs[n] * model_weights[n] for n in live_probs)
    live_is_confident = abs(live_ensemble_p - best_thr) >= confidence_margin
    live_side = "شراء" if live_ensemble_p >= best_thr else "بيع"
    live_conf = live_ensemble_p * 100 if live_side == "شراء" else (1 - live_ensemble_p) * 100

    # سياق القمم/القيعان/السيولة
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
# التشغيل
# ------------------------------------------------------------------------
if run_btn:
    status = st.empty()
    results = {}
    for cfg in CONFIGS:
        def cb(msg):
            status.info(f"⏳ {msg}")
        results[cfg["key"]] = analyze_config(cfg, years_back, use_walk_forward, wf_splits, cb)
    status.empty()
    st.session_state["results"] = results
    st.session_state["years_back"] = years_back

# ------------------------------------------------------------------------
# العرض
# ------------------------------------------------------------------------
if "results" in st.session_state:
    results = st.session_state["results"]
    tabs = st.tabs([f"⏱️ {cfg['label']} (Dead Zone {cfg['dead_zone_pct']}%)" for cfg in CONFIGS])

    for tab, cfg in zip(tabs, CONFIGS):
        with tab:
            r = results.get(cfg["key"], {})
            if r.get("error"):
                st.error(f"❌ {r['error']}")
                continue

            # --- التوصية الحية ---
            color = "🟢" if r["live_side"] == "شراء" else "🔴"
            st.markdown(f"### {color} التوصية الحالية: **{r['live_side']}** — ثقة {r['live_conf']:.1f}%")
            if not r["live_is_confident"]:
                st.warning("⚪ **منطقة حياد** — الموديل مش واثق بشكل كافٍ حالياً، فضّل الحذر أو الانتظار لإشارة أوضح.")
            cols = st.columns(len(r["live_probs"]))
            for i, (name, p) in enumerate(r["live_probs"].items()):
                side = "شراء" if p >= r["best_thr"] else "بيع"
                conf = p * 100 if side == "شراء" else (1 - p) * 100
                weight = r["model_weights"].get(name, 0.5)
                cols[i].metric(f"{name.upper()} (وزن {weight*100:.0f}%)", side, f"{conf:.1f}% ثقة")
            st.caption(f"آخر بيانات حتى: {r['last_ts']} | عتبة القرار المضبوطة: {r['best_thr']:.2f} "
                       f"| خصائص مستخدمة: {r['n_features_selected']}/{r['n_features_total']}")

            st.markdown("---")

            # --- دقة آخر 100 توقع ---
            st.markdown(f"### 📊 دقة آخر {r['n_used']} توقع (Ensemble): **{r['acc_last_n']:.1f}%**")
            if r["acc_last_n"] < 55:
                st.warning("⚠️ قريبة من العشوائية (50%) — ميزة الموديل بالإعداد ده ضعيفة حالياً.")
            elif r["acc_last_n"] >= 65:
                st.success("✅ أداء قوي نسبياً على المدى القريب.")

            if r["acc_confident_only"] is not None:
                st.markdown(
                    f"**🎯 دقة التوصيات الواثقة فقط** (استبعاد منطقة الحياد): "
                    f"**{r['acc_confident_only']:.1f}%** (على {r['n_confident']} من أصل {r['n_used']} توقع)"
                )
                st.caption(
                    "دي دقة القرارات اللي الموديل كان واثق فيها فعلاً وماكانش في منطقة حياد — "
                    "عادة أعلى من الدقة الإجمالية، لأنها بتستبعد الحالات الغامضة."
                )

            # --- Walk-Forward ---
            if r.get("wf_results"):
                st.markdown("### 🔁 استقرار الأداء (Walk-Forward)")
                wf_table = []
                for wf in r["wf_results"]:
                    row = {"الفترة #": wf["fold"], "من": wf["test_start"], "إلى": wf["test_end"]}
                    for name in ["lightgbm", "xgboost"]:
                        key = f"{name}_accuracy"
                        if key in wf:
                            row[f"Accuracy ({name})"] = round(wf[key], 3)
                    wf_table.append(row)
                st.dataframe(pd.DataFrame(wf_table), use_container_width=True, hide_index=True)

                for name in ["lightgbm", "xgboost"]:
                    key = f"{name}_accuracy"
                    accs = [wf[key] for wf in r["wf_results"] if key in wf]
                    if accs:
                        st.caption(f"**{name}**: متوسط {np.mean(accs)*100:.1f}% (± {np.std(accs)*100:.1f}% تذبذب) عبر {len(accs)} فترات")

            st.markdown("---")

            # --- سياق إضافي: قمم/قيعان/سيولة/ارتباط BTC ---
            st.markdown("### 🧭 سياق إضافي")
            ctx_lines = []
            if pd.notna(r.get("dist_swing_high")):
                ctx_lines.append(f"📍 السعر أبعد من آخر **قمة** بـ **{r['dist_swing_high']*100:.2f}%**")
            if pd.notna(r.get("dist_swing_low")):
                ctx_lines.append(f"📍 السعر أبعد من آخر **قاعدة** بـ **{r['dist_swing_low']*100:.2f}%**")
            if pd.notna(r.get("volume_zscore")):
                vz = r["volume_zscore"]
                if vz > 2:
                    ctx_lines.append("💧 **سيولة قوية دخلت السوق الآن**")
                elif vz < -1.5:
                    ctx_lines.append("💧 **السيولة ضعيفة حالياً**")
                else:
                    ctx_lines.append("💧 السيولة طبيعية")
            if pd.notna(r.get("corr_btc")):
                cb_val = r["corr_btc"]
                desc = "مرتفع (تابع للسوق)" if abs(cb_val) > 0.6 else ("متوسط" if abs(cb_val) > 0.3 else "منخفض (مستقل)")
                ctx_lines.append(f"📎 ارتباط بـ BTC: **{cb_val:.2f}** — {desc}")
            for line in ctx_lines:
                st.write(line)

    st.markdown("---")
    st.warning("⚠️ توقعات إحصائية تعليمية وليست نصيحة استثمارية. الأداء الفعلي قد يختلف تماماً.")
else:
    st.info("👈 اضبط الإعدادات من الشريط الجانبي، وبعدين دوس «تحديث وتشغيل التحليل الكامل».")
    st.caption("⚠️ التحليل ده أثقل من النسخة الخفيفة (بيحلل إعدادين × موديلين × Walk-Forward)، فممكن ياخد وقت أطول.")
