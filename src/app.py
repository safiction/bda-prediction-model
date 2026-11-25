"""Simple local Streamlit dashboard. Heavy stages run in src.worker, never in a render."""

import os
import sys
from collections import deque
from datetime import date, timedelta
from pathlib import Path

# Streamlit запускает файл как скрипт; корень проекта нужен для импортов src.*.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from src.features import KEY_COLUMNS
from src.final import completed_models, read_json, snapshot_details
from src.predict import load_model, predict_probabilities
from src.worker import actions, current_job, load_config, replay_day, submit_job


PURPLE, ORANGE = "#7700FF", "#FF4F12"
CONFIG_PATH = Path(os.environ.get("BDA_CONFIG", "config.json")).resolve()
CONFIG = load_config(CONFIG_PATH)
STATE, ARTIFACTS = Path(CONFIG["state_dir"]), Path(CONFIG["artifacts_dir"])
MODEL_NAMES = {"logistic": "Logistic Regression", "xgboost": "XGBoost"}
STAGES = {"queued": "В очереди", "ingest": "Загрузка", "clean": "Очистка", "snapshots": "Подготовка снимков",
          "save": "Запись Parquet", "aggregate": "Дневные агрегаты",
          "features": "Признаки", "labels": "Метки", "train": "Обучение", "freeze": "Фиксация прогнозов",
          "evaluate": "Финальная оценка", "completed": "Завершено", "failed": "Ошибка", "interrupted": "Прервано"}


def number(value, decimals=3):
    return "Нет данных" if value is None or pd.isna(value) else f"{value:,.{decimals}f}".replace(",", " ")


def card(column, label, value):
    with column.container(border=True):
        st.metric(label, value)


def launch(action):
    try:
        submit_job(action, CONFIG_PATH)
        st.rerun()
    except (ValueError, OSError) as error:
        st.error(str(error))


def latest_job():
    paths = list((STATE / "jobs").glob("*.json"))
    records = [read_json(path) for path in paths if path.name != "active.json"]
    return max(records, key=lambda job: job["created_timestamp"]) if records else None


@st.fragment(run_every="2s")
def pipeline_view():
    job = current_job(CONFIG)
    last = job or latest_job()
    previous = st.session_state.get("last_job_status")
    status = last["status"] if last else None
    st.session_state["last_job_status"] = status
    if previous in ("queued", "running") and status in ("completed", "failed"):
        st.rerun()
    ingest = read_json(STATE / "ingest.json", {})
    clock = replay_day(CONFIG)
    st.header("Данные и pipeline")
    st.caption("Последовательная обработка исторических данных. Обновление страницы не запускает новое задание.")
    columns = st.columns(3)
    card(columns[0], "День replay", str(clock))
    card(columns[1], "Доступность данных", ingest.get("available_through", "Ещё не загружены"))
    active = read_json(STATE / "active_model.json", {})
    metadata = read_json(ARTIFACTS / "models" / active.get("version", "none") / "metadata.json", {})
    card(columns[2], "Активная модель", MODEL_NAMES.get(metadata.get("model_name"), "Пока нет"))
    if active:
        st.caption(f"Версия: {active['version']}")
    if job:
        st.info(f"Выполняется: {STAGES.get(job['stage'], job['stage'])}")
        if job.get("message"):
            st.caption(job["message"])
        if job.get("files_processed"):
            st.write(f"Обработано файлов: **{job['files_processed']}**")
    elif last and last["status"] == "failed":
        st.error(last["error"])
        st.caption("Последний успешный результат сохранён. Проверьте журнал и повторите доступное действие.")
    elif last:
        st.success("Последнее задание завершено")

    options = actions(CONFIG)
    busy = job is not None or (STATE / "ingest.lock").exists()
    st.subheader("Следующий шаг")
    left, right = st.columns(2)
    for column, action in ((left, "prepare"), (right, "train")):
        if column.button(options[action]["label"], key=action, type="primary" if action == "prepare" else "secondary",
                         disabled=busy or not options[action]["enabled"], use_container_width=True):
            launch(action)
    if options["snapshots"]["enabled"] and not options["train"]["ready"]:
        if st.button("Подготовить снимки текущей партии", disabled=busy):
            launch("snapshots")
    if clock >= CONFIG["training_replay_days"][-1] + CONFIG["label_delivery_lag_days"]:
        left, right = st.columns(2)
        for column, action in ((left, "freeze"), (right, "evaluate")):
            if column.button(options[action]["label"], key=action, disabled=busy or not options[action]["enabled"], use_container_width=True):
                launch(action)
    st.caption("Обучение: этапы 60 / 67 / 74. Фиксация прогнозов: день 76. Финальные исходы: день 91 с учётом доставки.")

    if ingest:
        st.subheader("Качество и полнота")
        rows = []
        for source in ("cdr", "tickets"):
            quality = ingest["quality"][source]
            rows.append({"Источник": source.upper(), "Входных строк": quality.get("input_rows", 0),
                         "Строк в Parquet": quality.get("output_rows", 0),
                         "Удалено дубликатов": quality.get("duplicate_rows_removed", 0),
                         "Повреждённые CSV": quality.get("dropped_malformed", 0),
                         "Некорректный ключ / дата": quality.get("dropped_invalid_identity_or_timestamp", 0),
                         "Ожидают доставки": quality.get("pending_availability", 0)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"События загружены по {ingest['loaded_event_end']}. Полных дней CDR: {len(ingest['complete_cdr_dates'])}; tickets: {len(ingest['complete_ticket_dates'])}.")
        with st.expander("Полнота по дням"):
            dates = sorted(set(ingest["complete_cdr_dates"]) | set(ingest["complete_ticket_dates"]) | {ingest["loaded_event_end"]})
            st.dataframe(pd.DataFrame({"Дата": dates, "CDR": [value in ingest["complete_cdr_dates"] for value in dates],
                                      "Tickets": [value in ingest["complete_ticket_dates"] for value in dates]}), hide_index=True, use_container_width=True)
    if last:
        with st.expander("Журнал последнего задания", expanded=last["status"] == "failed"):
            path = STATE / "jobs" / f"{last['job_id']}.log"
            if path.exists():
                with path.open(encoding="utf-8", errors="replace") as stream:
                    lines = list(deque(stream, maxlen=40))
                st.code("".join(lines), language="text")
            else:
                st.caption("Журнал появится после запуска обработчика")
    if st.button("Обновить состояние", key="refresh"):
        st.rerun()


def history_view():
    st.header("История моделей")
    active = read_json(STATE / "active_model.json", {}).get("version")
    metadata = completed_models(CONFIG)
    if not metadata:
        st.info("Модели появятся после первого успешного обучения")
        return
    if active in metadata:
        selected = metadata[active]
        st.subheader(MODEL_NAMES[selected["model_name"]])
        st.caption(f"Активная версия: {active}")
        split = selected["split"]
        st.caption(f"Train: {', '.join(row['snapshot_date'] for row in split['training'])}. Validation обучения: {split['validation']['snapshot_date']}.")
        columns = st.columns(3)
        for column, name, label in zip(columns, ("pr_auc", "log_loss", "recall"), ("PR-AUC · AP", "Log loss", "Recall")):
            card(column, label, number(selected["metrics"][name]))
    rows, chart_rows = [], []
    for path in sorted((ARTIFACTS / "runs").glob("*/run.json")):
        run = read_json(path)
        if run["status"] != "completed":
            continue
        validation = run["split"]["validation"]
        entries = [(version, item, item["metrics"], "Новая модель") for version, item in run["candidates"].items()]
        if run["incumbent"]:
            version = run["parent_version"]
            entries.append((version, metadata[version], run["incumbent"]["metrics"], "Повторная оценка"))
        for version, item, metrics, role in entries:
            row = {"Модель": MODEL_NAMES[item["model_name"]], "Роль": role, "Версия": version,
                   "Активная": version == active, "Validation": validation["snapshot_date"],
                   "Исходы": f"{validation['outcome_start']} — {validation['outcome_end']}",
                   "Train строк": item["rows"]["training"], "Validation строк": run["rows"]["validation"],
                   "Доля неактивных": run["class_counts"]["validation"]["1"] / run["rows"]["validation"],
                   "Обучение, с": item["fit_seconds"], **metrics}
            rows.append(row)
            if role == "Новая модель":
                chart_rows.append({"Модель": row["Модель"], "Дата validation": validation["snapshot_date"], **metrics})
    st.caption("PR-AUC — average precision. Окна validation меняются: оценки разных дат относятся к разным наблюдениям.")
    charts = pd.DataFrame(chart_rows)
    for column, metric, label in zip(st.columns(2), ("pr_auc", "log_loss"), ("PR-AUC по датам validation", "Log loss по датам validation")):
        with column:
            st.subheader(label)
            chart = alt.Chart(charts).mark_line(point=True).encode(
                x=alt.X("Дата validation:T", title="Дата validation", axis=alt.Axis(format="%d.%m.%Y", tickCount=max(1, charts["Дата validation"].nunique()))),
                y=alt.Y(f"{metric}:Q", title=metric),
                color=alt.Color("Модель:N", scale=alt.Scale(domain=list(MODEL_NAMES.values()), range=[PURPLE, ORANGE])),
                tooltip=["Модель:N", "Дата validation:T", alt.Tooltip(f"{metric}:Q", format=".6f")]).properties(height=235)
            st.altair_chart(chart, use_container_width=True)
    frame = pd.DataFrame(rows)
    columns = ["Модель", "Validation", "pr_auc", "log_loss", "roc_auc", "precision", "recall", "f1",
               "Роль", "Активная", "Исходы", "Train строк", "Validation строк", "Доля неактивных", "Обучение, с", "Версия"]
    st.dataframe(frame[columns], hide_index=True, use_container_width=True,
                 column_config={name: st.column_config.NumberColumn(format="%.4f")
                                for name in ("pr_auc", "log_loss", "roc_auc", "precision", "recall", "f1", "Доля неактивных", "Обучение, с")})
    with st.expander("Confusion matrix"):
        st.dataframe(pd.DataFrame([{"Версия": row["Версия"], "Validation": row["Validation"], **row["confusion_matrix"]} for row in rows]), hide_index=True, use_container_width=True)
    final = read_json(ARTIFACTS / "final_evaluation/comparison.json")
    st.subheader("Общая финальная когорта")
    if final:
        st.caption(f"Снимок {final['snapshot_date']}; исходы {final['outcome_start']} — {final['outcome_end']}. Модель была выбрана до раскрытия исходов.")
        st.dataframe(pd.DataFrame(final["models"]).drop(columns="confusion_matrix"), hide_index=True, use_container_width=True)
    else:
        st.info("Исходы ожидаются. Финальное сравнение появится после фиксации на дне 76 и доставки исходов к дню 91.")


@st.cache_data(show_spinner=False)
def eligible_ids(path):
    return sorted(pd.read_parquet(path, columns=["subscriber_key"]).subscriber_key.tolist())


@st.cache_resource(show_spinner=False)
def cached_model(path):
    return load_model(path)


def prediction_view():
    st.header("Прогноз абонента")
    versions = completed_models(CONFIG)
    snapshots = read_json(STATE / "snapshots.json", {}).get("snapshots", [])
    if not versions or not snapshots:
        st.info("Сначала подготовьте снимки и обучите модель")
        return
    frozen = read_json(ARTIFACTS / "final_predictions/manifest.json")
    if frozen:
        options = [{"snapshot_day": frozen["snapshot_day"], "snapshot_date": frozen["snapshot_date"]}]
    else:
        options = snapshots
    selected = st.selectbox("Исторический снимок", options, index=len(options) - 1,
                            format_func=lambda row: f"День {row['snapshot_day']} · {row['snapshot_date']}", key="snapshot")
    day = selected["snapshot_day"]
    if frozen:
        features_path = ARTIFACTS / "final_predictions/features.parquet"
        versions = {version: versions[version] for version in frozen["thresholds"]}
        st.caption("Зарезервированная финальная когорта: используются зафиксированные прогнозы дня 76.")
        snapshot = frozen
        folder = None
    else:
        folder, snapshot = snapshot_details(CONFIG, day)
        features_path = folder / "features"
    active = read_json(STATE / "active_model.json", {})["version"]
    choices = list(versions)
    version = st.selectbox("Версия модели", choices, index=choices.index(active) if active in choices else 0,
        format_func=lambda value: f"{MODEL_NAMES[versions[value]['model_name']]} · этап {versions[value]['milestone']}" + (" · активная" if value == active else ""), key="model")
    st.caption(f"Версия: {version}")
    ids = eligible_ids(str(features_path))
    query = st.text_input("Поиск по ID", placeholder="Например, SUB000000000", key="query").strip().upper()
    matches = [value for value in ids if value.startswith(query)][:100]
    if not matches:
        st.info("В выбранном снимке нет подходящего активного абонента")
        return
    subscriber = st.selectbox("Допущенный абонент", matches, key="subscriber")
    st.caption(f"Показано до 100 совпадений из {len(ids):,} допущенных абонентов. Уточните ID для поиска.".replace(",", " "))
    context = (str(features_path), version, subscriber)
    if st.button("Рассчитать вероятность", type="primary", key="predict"):
        features = pd.read_parquet(features_path, filters=[("subscriber_key", "==", subscriber)])
        if len(features) != 1:
            raise ValueError("Абонент должен быть уникальным в выбранном снимке")
        if frozen:
            saved = pd.read_parquet(ARTIFACTS / "final_predictions" / f"{version}.parquet", filters=[("subscriber_key", "==", subscriber)]).iloc[0]
            probability, threshold = float(saved.probability), float(saved.threshold)
        else:
            pipeline, metadata = cached_model(str(ARTIFACTS / "models" / version))
            probability = float(predict_probabilities(pipeline, features.drop(columns=KEY_COLUMNS), metadata["feature_names"])[0])
            threshold = metadata["threshold"]
        st.session_state["prediction"] = {"context": context, "probability": probability, "threshold": threshold, "features": features.iloc[0].to_dict()}
    result = st.session_state.get("prediction")
    if not result or result["context"] != context:
        return
    probability, threshold = result["probability"], result["threshold"]
    left, right = st.columns([1, 2])
    with left.container(border=True):
        st.metric("Вероятность 14-дневной неактивности", f"{probability:.1%}")
        st.write("Прогноз: **неактивность**" if probability >= threshold else "Прогноз: **активность**")
        st.caption(f"Порог {threshold:.0%} · снимок {selected['snapshot_date']}")
    with right.container(border=True):
        row = result["features"]
        display = {"Событий": number(row["event_count"], 0), "Дней с активностью": number(row["active_days"], 0),
                   "Дней с последней активности": number(row["days_since_last_activity"], 0),
                   "Голосовых вызовов": number(row["voice_count"], 0), "Средний звонок, с": number(row["avg_voice_duration_sec"], 1),
                   "Загрузка, МБ": number(row["total_bytes_down"] / 1024 ** 2, 1),
                   "Выгрузка, МБ": number(row["total_bytes_up"] / 1024 ** 2, 1), "Обращений": number(row["ticket_count"], 0)}
        st.dataframe(pd.DataFrame(display.items(), columns=["Исторический признак", "Значение"]), hide_index=True, use_container_width=True)
    final_outcomes = ARTIFACTS / "final_evaluation/outcomes.parquet"
    labels_path = final_outcomes if frozen else folder / "labels"
    mature = final_outcomes.exists() if frozen else snapshot["label_status"] == "mature" and not snapshot["missing_outcome_dates"]
    if mature:
        observed = pd.read_parquet(labels_path, filters=[("subscriber_key", "==", subscriber)])
        target = observed.iloc[0].target
        if pd.notna(target):
            st.info(f"Наблюдаемый исход {snapshot['outcome_start']} — {snapshot['outcome_end']}: " + ("CDR-активности не было" if target == 1 else "CDR-активность была"))
    else:
        st.info("Исход пока неизвестен: полное 14-дневное окно ещё недоступно.")
    with st.expander("Все признаки"):
        st.dataframe(pd.DataFrame([{"Признак": name, "Значение": result["features"][name]} for name in CONFIG["feature_allowlist"]]), hide_index=True, use_container_width=True)


st.set_page_config(page_title="Прогноз неактивности", page_icon="◈", layout="wide")
st.markdown("""<style>
.stApp {background:radial-gradient(ellipse at top right, #221033 0, #0d0a12 50%);}
.block-container {max-width:1180px; padding-top:2rem; padding-bottom:3rem;}
[data-testid="stSidebar"] {background:#15101c; border-right:1px solid #2e203d;}
[data-testid="stHeader"] {background:transparent; height:0;}
[data-testid="stDecoration"] {display:none;}
[data-testid="stToolbar"] {display:none;}
h1 {font-weight:750!important; letter-spacing:-.04em; font-size:2.9rem!important;}
h2 {letter-spacing:-.025em;}
[data-testid="stMetricValue"] {font-size:1.65rem;}
.stButton button {border-radius:10px; min-height:44px;}
.brand {border-left:5px solid #7700ff; padding:0 0 0 14px; margin:10px 0 26px;}
.brand b {font-size:1.15rem;} .brand span {display:block; color:#ff6c3a; font-size:.9rem; margin-top:5px;}
.eyebrow {color:#ff794a; font-size:.9rem; letter-spacing:.12em; font-weight:650;}
</style>""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="brand"><b>Неактивность абонентов</b><span>Учебный проект · BDA</span></div>', unsafe_allow_html=True)
    section = st.radio("Раздел", ["Данные и pipeline", "История моделей", "Прогноз абонента"], label_visibility="collapsed", key="section")
    st.divider()
    st.caption("CDR + tickets\n\nОкно признаков: 14 дней\n\nГоризонт прогноза: 14 дней")
st.markdown('<div class="eyebrow">CDR & TICKETS · HISTORICAL REPLAY</div>', unsafe_allow_html=True)
st.title("Прогноз неактивности")
st.caption("14-day inactivity proxy for churn · косвенный показатель оттока по отсутствию CDR-активности")
try:
    if section == "Данные и pipeline":
        pipeline_view()
    elif section == "История моделей":
        history_view()
    else:
        prediction_view()
except (ValueError, OSError, KeyError) as error:
    st.error(str(error))
    st.caption("Проверьте состояние pipeline и журнал задания в разделе данных.")
