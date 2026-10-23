"""Review model errors clip by clip, from a bbeedeep cross-validation study.

Input: the `collected/` folder of a study (bbeedeep: scripts/collect_cv.py, then
scripts/aggregate_predictions.py --task <task>), which holds `review_<task>.parquet`: one
row per clip with every model's held-out prediction and class probabilities. For each clip:
spectrogram, audio, target vs predictions, per-model probabilities, metadata. Verdicts are
written on each click to a CSV (keyed by clip file name and task) and reloaded on start.

    streamlit run ErrorReview.py
"""

import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import audio_utils

st.set_page_config(page_title="BumbleBuzz error review", layout="wide", initial_sidebar_state="expanded")

CLASS_NAMES = {
    "buzz_type": {"F": "Flight", "C": "Stress (constrained)", "E": "Escape", "U": "Tube", "P": "Sonication"},
}
VERDICTS = {
    "label_ok": "✅ Label correct",
    "mislabeled": "❌ Mislabeled",
    "ambiguous": "❓ Ambiguous",
    "bad_audio": "🔇 Bad audio",
}
VERDICT_COLUMNS = ["cut_filename", "task", "label", "verdict", "corrected_label", "majority_pred",
                   "n_wrong", "n_models", "suspicion", "note", "reviewer", "time"]
META_COLUMNS = ["Species", "Cast", "Buzz_type", "sound.files", "Protocol_Type", "Country", "Date",
                "duration", "Buzz_condition"]
GROUP_COLUMNS = ["sound.files", "Species", "Cast", "Protocol_Type", "Country", "Buzz_condition", "Date"]


# -- data ---------------------------------------------------------------------------------

@st.cache_data
def load_review(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def load_verdicts(path: str) -> pd.DataFrame:
    if path and os.path.exists(path):
        return pd.read_csv(path, dtype={"note": str, "corrected_label": str})
    return pd.DataFrame(columns=VERDICT_COLUMNS)


def save_verdict(path: str, row: dict) -> None:
    """Upsert one verdict, keyed by (cut_filename, task)."""
    verdicts = load_verdicts(path)
    keep = ~((verdicts.cut_filename == row["cut_filename"]) & (verdicts.task == row["task"]))
    verdicts = pd.concat([verdicts[keep], pd.DataFrame([row])], ignore_index=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    verdicts[VERDICT_COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, path)


def setups_of(df: pd.DataFrame) -> list:
    return sorted(c[: -len("__pred")] for c in df.columns if c.endswith("__pred"))


def classes_of(df: pd.DataFrame) -> list:
    return [c[len("mean__p_"):] for c in df.columns if c.startswith("mean__p_")]


def recompute_consensus(df: pd.DataFrame, setups: list, classes: list) -> pd.DataFrame:
    """Majority vote / agreement over the selected setups only."""
    df = df.copy()
    known = df["target"].notna()
    preds = df[[f"{s}__pred" for s in setups]]
    df["n_models"] = preds.notna().sum(axis=1)
    df["n_wrong"] = preds.ne(df["target"], axis=0).where(known).sum(axis=1)
    mean_p = np.stack([df[[f"{s}__p_{c}" for s in setups]].mean(axis=1).to_numpy() for c in classes], axis=1)
    votes = np.stack([(preds == c).sum(axis=1).to_numpy() for c in classes], axis=1)
    df["majority_pred"] = np.array(classes)[(votes + 1e-3 * mean_p).argmax(axis=1)]
    df["n_agree_majority"] = preds.eq(df["majority_pred"], axis=0).sum(axis=1)
    df["mean_p_target"] = df[[f"{s}__p_target" for s in setups]].mean(axis=1)
    df["suspicion"] = 1 - df["mean_p_target"]
    for j, c in enumerate(classes):
        df[f"sel__p_{c}"] = mean_p[:, j]
    return df


def label(task: str, c) -> str:
    name = CLASS_NAMES.get(task, {}).get(c)
    return f"{c} ({name})" if name else str(c)


# -- sidebar ------------------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Inputs")
    collected = st.text_input("Collected study folder", placeholder=".../logs/cv/model_selection/collected",
                              help="Folder with review_<task>.parquet (scripts/aggregate_predictions.py)")
    audio_folder = st.text_input("Audio folder", placeholder=".../data/Cuts_Database/Cuts_Database_ALL_SPECIES")
    review_files = sorted(Path(collected).glob("review_*.parquet")) if collected and os.path.isdir(collected) else []
    if not review_files and collected and os.path.isdir(collected):
        review_files = sorted(Path(collected).glob("review_*.csv"))
    task = st.selectbox("Task", [f.stem[len("review_"):] for f in review_files]) if review_files else None
    verdict_path = st.text_input("Verdicts CSV (outside any repo)",
                                 value=str(Path.home() / "bumblebuzz_reviews" / f"{task or 'task'}_verdicts.csv"))
    reviewer = st.text_input("Reviewer", value=os.environ.get("USER", ""))

if not (review_files and task and audio_folder and os.path.isdir(audio_folder)):
    st.title("🐝 BumbleBuzz error review")
    st.info(
        "1. **Collected study folder**: the `collected/` folder of a bbeedeep CV study, after "
        "`scripts/aggregate_predictions.py --task buzz_type`.\n"
        "2. **Audio folder**: where the clips live (`Cuts_Database_ALL_SPECIES`).\n"
        "3. **Verdicts CSV**: where your verdicts are saved (created if missing)."
    )
    if collected and not review_files:
        st.warning("No review_<task>.parquet in that folder: run scripts/aggregate_predictions.py first.")
    if audio_folder and not os.path.isdir(audio_folder):
        st.error("The audio folder does not exist.")
    st.stop()

review_path = str(next(f for f in review_files if f.stem == f"review_{task}"))
data = load_review(review_path)
all_setups = setups_of(data)
classes = classes_of(data)
verdicts = load_verdicts(verdict_path)
verdicts = verdicts[verdicts.task == task].set_index("cut_filename")

with st.sidebar:
    st.header("🤖 Models")
    setups = st.multiselect("Models used for agreement", all_setups, default=all_setups)
    if not setups:
        st.stop()
    df = recompute_consensus(data, setups, classes)
    df["reviewed"] = df.cut_filename.isin(verdicts.index)

    st.header("🔎 Queue")
    show = st.radio("Clips", ["Majority vote wrong", "At least one model wrong",
                              "All models wrong, same class", "All clips"], index=0)
    min_wrong = st.slider("Min. models wrong", 0, len(setups), 0)
    targets = st.multiselect("Target class", classes, format_func=lambda c: label(task, c))
    majority = st.multiselect("Majority prediction", classes, format_func=lambda c: label(task, c))
    species = st.multiselect("Species", sorted(df.Species.dropna().unique())) if "Species" in df else []
    recording = st.text_input("Recording (sound.files) contains")
    hide_reviewed = st.checkbox("Hide reviewed clips", value=True)
    sort_by = st.selectbox("Sort by", ["suspicion (most suspicious first)", "models wrong", "file name"])

    st.header("🎛️ Spectrogram")
    with st.expander("Parameters", expanded=False):
        mel = st.checkbox("Mel scale", value=False)
        n_fft = st.slider("n_fft", 256, 4096, 2048, 256)
        hop_length = st.slider("Hop length", 64, 1024, 256, 64)
        db_range = st.slider("dB range", 30, 120, 80)
        fmax = st.number_input("Max frequency (Hz, 0 = Nyquist)", 0, 48000, 0, 500)

# -- queue --------------------------------------------------------------------------------

known = df.target.notna()
q = df[known]
if show == "Majority vote wrong":
    q = q[q.majority_pred != q.target]
elif show == "At least one model wrong":
    q = q[q.n_wrong > 0]
elif show == "All models wrong, same class":
    q = q[(q.n_wrong == len(setups)) & (q.n_agree_majority == len(setups))]
q = q[q.n_wrong >= min_wrong]
if targets:
    q = q[q.target.isin(targets)]
if majority:
    q = q[q.majority_pred.isin(majority)]
if species:
    q = q[q.Species.isin(species)]
if recording:
    q = q[q["sound.files"].astype(str).str.contains(recording, case=False, regex=False)]
if hide_reviewed:
    q = q[~q.reviewed]
q = {"suspicion (most suspicious first)": q.sort_values("suspicion", ascending=False),
     "models wrong": q.sort_values(["n_wrong", "suspicion"], ascending=False),
     "file name": q.sort_values("cut_filename")}[sort_by].reset_index(drop=True)

review_tab, summary_tab = st.tabs(["🎧 Review", "📊 Summary"])

# -- review -------------------------------------------------------------------------------

with review_tab:
    st.markdown(f"### 🐝 {task}: {len(q)} clips in the queue "
                f"({int(df.reviewed.sum())} reviewed in total, {len(setups)} models)")
    search = st.text_input("🔍 Jump to a clip by file name (any clip, not only the queue)")
    if "pos" not in st.session_state:
        st.session_state.pos = 0
    if search:
        hits = df[df.cut_filename.str.contains(search, case=False, regex=False)]
        if hits.empty:
            st.warning("No clip matches that name.")
            st.stop()
        name = st.selectbox(f"{len(hits)} match(es)", hits.cut_filename.tolist()) if len(hits) > 1 else hits.cut_filename.iloc[0]
        row = df[df.cut_filename == name].iloc[0]
    else:
        if q.empty:
            st.success("Nothing left in this queue. Change the filters, or untick 'Hide reviewed clips'.")
            st.stop()
        st.session_state.pos = min(st.session_state.pos, len(q) - 1)
        nav = st.columns([1, 1, 1, 3])
        if nav[0].button("⬅️ Previous", use_container_width=True):
            st.session_state.pos = (st.session_state.pos - 1) % len(q)
            st.rerun()
        if nav[1].button("➡️ Next", use_container_width=True):
            st.session_state.pos = (st.session_state.pos + 1) % len(q)
            st.rerun()
        if nav[2].button("🎲 Random", use_container_width=True):
            st.session_state.pos = int(np.random.randint(len(q)))
            st.rerun()
        nav[3].markdown(f"**Clip {st.session_state.pos + 1} of {len(q)}**")
        row = q.iloc[st.session_state.pos]

    name = row.cut_filename
    audio_path = os.path.join(audio_folder, name)
    wrong = row.majority_pred != row.target
    st.code(name, language=None)
    head = st.columns(4)
    head[0].metric("Label (target)", label(task, row.target))
    head[1].metric("Majority prediction", label(task, row.majority_pred), delta="disagrees" if wrong else "agrees",
                   delta_color="inverse" if wrong else "normal")
    head[2].metric("Models wrong", f"{int(row.n_wrong)} / {int(row.n_models)}")
    head[3].metric("Mean p(label)", f"{row.mean_p_target:.2f}")
    if name in verdicts.index:
        v = verdicts.loc[name]
        extra = f" → {v.corrected_label}" if isinstance(v.corrected_label, str) and v.corrected_label else ""
        st.info(f"Already reviewed: **{VERDICTS.get(v.verdict, v.verdict)}{extra}** by {v.reviewer} ({v.time})"
                + (f" — {v.note}" if isinstance(v.note, str) and v.note else ""))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("🎵 Audio")
        if os.path.exists(audio_path):
            st.audio(audio_path)
        else:
            st.error(f"Audio file not found: {audio_path}")

        st.subheader("🤖 Models")
        models = pd.DataFrame({
            "model": setups,
            "prediction": [label(task, row[f"{s}__pred"]) for s in setups],
            "p(label)": [row[f"{s}__p_target"] for s in setups],
            "p(prediction)": [row[f"{s}__p_pred"] for s in setups],
            "seed agreement": [row.get(f"{s}__seed_agreement", np.nan) for s in setups],
            "correct": [row[f"{s}__pred"] == row.target for s in setups],
        })
        st.dataframe(models, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["p(label)", "p(prediction)", "seed agreement"]})

        probs = pd.DataFrame({s: [row[f"{s}__p_{c}"] for c in classes] for s in setups}, index=classes)
        fig, ax = plt.subplots(figsize=(6, 2.8))
        width = 0.8 / len(setups)
        x = np.arange(len(classes))
        for i, s in enumerate(setups):
            ax.bar(x + i * width - 0.4 + width / 2, probs[s], width, label=s)
        ax.set_xticks(x, [label(task, c) for c in classes], rotation=20, ha="right", fontsize=8)
        ax.axvspan(classes.index(row.target) - 0.45, classes.index(row.target) + 0.45, color="green", alpha=0.08)
        ax.set_ylim(0, 1)
        ax.set_ylabel("probability")
        ax.set_title("Class probabilities per model (label shaded)", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("📊 Spectrogram")
        if os.path.exists(audio_path):
            try:
                audio, sr = audio_utils.load_audio(audio_path, duration=None)
                st.image(audio_utils.create_spectrogram(audio, sr, n_fft, hop_length, db_range, mel=mel,
                                                        fmax=fmax or None, title=f"{len(audio) / sr:.2f} s @ {sr} Hz"))
            except Exception as e:  # noqa: BLE001 -- show any decoding error in the page
                st.error(f"Could not load audio: {e}")
        meta = {c: row[c] for c in META_COLUMNS if c in row.index and pd.notna(row[c])}
        st.subheader("🏷️ Metadata")
        st.dataframe(pd.DataFrame(meta.items(), columns=["field", "value"]).astype(str), hide_index=True,
                     use_container_width=True)

    st.subheader("🎯 Verdict")
    vcols = st.columns([1, 1, 1, 1])
    corrected = st.selectbox("Correct class (for 'Mislabeled')", classes,
                             index=classes.index(row.majority_pred), format_func=lambda c: label(task, c))
    note = st.text_input("Note (optional)", key=f"note_{name}")
    for col, (key, text) in zip(vcols, VERDICTS.items()):
        if col.button(text, use_container_width=True, key=f"verdict_{key}"):
            save_verdict(verdict_path, {
                "cut_filename": name, "task": task, "label": row.target, "verdict": key,
                "corrected_label": corrected if key == "mislabeled" else "",
                "majority_pred": row.majority_pred, "n_wrong": int(row.n_wrong), "n_models": int(row.n_models),
                "suspicion": round(float(row.suspicion), 4), "note": note, "reviewer": reviewer,
                "time": datetime.now().isoformat(timespec="seconds"),
            })
            # With reviewed clips hidden, the next clip moves into this position.
            if not search and not hide_reviewed:
                st.session_state.pos += 1
            st.rerun()
    st.caption(f"Verdicts are saved to {verdict_path}")

# -- summary ------------------------------------------------------------------------------

with summary_tab:
    k = df[known]
    wrong_major = k.majority_pred != k.target
    all_same = (k.n_wrong == len(setups)) & (k.n_agree_majority == len(setups))
    cols = st.columns(4)
    cols[0].metric("Clips with a label", len(k))
    cols[1].metric("Majority vote wrong", f"{int(wrong_major.sum())} ({wrong_major.mean():.1%})")
    cols[2].metric("All models wrong, same class", int(all_same.sum()))
    cols[3].metric("Reviewed", int(k.reviewed.sum()))

    st.subheader("Majority vote vs label (rows = label)")
    st.dataframe(pd.crosstab(k.target.map(lambda c: label(task, c)), k.majority_pred.map(lambda c: label(task, c))),
                 use_container_width=True)

    st.subheader("Clips by number of models wrong")
    st.bar_chart(k.n_wrong.value_counts().sort_index())

    st.subheader("Where the errors concentrate")
    gcol = st.selectbox("Group by", [c for c in GROUP_COLUMNS if c in k])
    min_n = st.slider("Min. clips per group", 1, 100, 20)
    groups = k.assign(majority_wrong=wrong_major, all_wrong_same=all_same).groupby(gcol).agg(
        clips=("cut_filename", "size"), majority_wrong=("majority_wrong", "mean"),
        all_wrong_same=("all_wrong_same", "sum"), mean_suspicion=("suspicion", "mean"))
    groups = groups[groups.clips >= min_n].sort_values("majority_wrong", ascending=False)
    st.dataframe(groups, use_container_width=True,
                 column_config={"majority_wrong": st.column_config.NumberColumn("error rate", format="%.3f"),
                                "mean_suspicion": st.column_config.NumberColumn(format="%.3f")})

    if len(verdicts):
        st.subheader("Verdicts so far")
        v = verdicts.reset_index()
        st.dataframe(pd.crosstab(v.label, v.verdict), use_container_width=True)
        st.download_button("💾 Download verdicts CSV", v.to_csv(index=False), file_name=f"{task}_verdicts.csv",
                           mime="text/csv")
