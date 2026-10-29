import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE SETUP
# --------------------------------------------------

st.set_page_config(
    page_title="Waste Notes",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# STYLING
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;1,500&display=swap');

:root {
    --paper: #f7f6f0;
    --ink: #293c30;
    --green: #4b6951;
    --muted: #788276;
    --line: #dcded4;
    --sage: #e9ede3;
    --accent: #bd7855;
}

.stApp {
    background: var(--paper);
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1120px;
    padding-top: 2.3rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: var(--ink) !important;
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 500 !important;
}

p, label, input, button {
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #eeefe7;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

.brand {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--green);
    margin-bottom: 20px;
}

.hero {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(42px, 6vw, 72px);
    line-height: 1.06;
    letter-spacing: -1.8px;
    color: var(--ink);
    margin: 12px 0 18px 0;
}

.hero em {
    color: var(--green);
    font-weight: 500;
}

.intro {
    max-width: 630px;
    color: #6c786b;
    font-size: 15px;
    line-height: 1.9;
}

.eyebrow {
    color: var(--green);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 10px;
}

.rule {
    height: 1px;
    background: var(--line);
    margin: 30px 0;
}

.section-note {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.7;
    margin-top: -5px;
    margin-bottom: 20px;
}

.stButton > button,
.stDownloadButton > button {
    background: var(--green);
    color: white;
    border: 1px solid var(--green);
    border-radius: 2px;
    padding: 0.65rem 1rem;
    min-height: 45px;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #344d3a;
    color: white;
    border-color: #344d3a;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
    background: #fffef9;
    border-color: #d7dbd0;
    border-radius: 2px;
}

[data-testid="stForm"] {
    border: none;
    background: transparent;
    padding: 0;
}

[data-testid="stMetric"] {
    background: transparent;
    border: none;
    padding: 0;
}

.result {
    background: var(--sage);
    border-left: 3px solid var(--green);
    padding: 25px 28px;
    margin: 18px 0 24px 0;
}

.result-label {
    font-size: 11px;
    letter-spacing: 1.7px;
    text-transform: uppercase;
    color: #647463;
}

.result-number {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(38px, 5vw, 58px);
    color: #2d4934;
    margin: 10px 0 5px 0;
    line-height: 1.2;
}

.result-caption {
    color: #657364;
    font-size: 13px;
}

.footer {
    border-top: 1px solid var(--line);
    margin-top: 42px;
    padding-top: 18px;
    font-size: 12px;
    color: #81887c;
    line-height: 1.8;
}

[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

/* Keep all regular text readable even if Streamlit is using a dark theme. */
.stApp,
.stApp p,
.stApp span,
.stApp li,
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label {
    color: var(--ink);
}

/* Text entered into fields and the selected value in dropdowns. */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] input,
div[data-baseweb="select"] [role="combobox"],
div[data-baseweb="select"] [data-testid="stMarkdownContainer"],
div[data-baseweb="select"] span {
    color: #293c30 !important;
    -webkit-text-fill-color: #293c30 !important;
    background-color: #fffef9;
}

/* Dropdown menu options, including options rendered in a portal. */
[role="listbox"],
[role="listbox"] *,
[role="option"],
[role="option"] * {
    color: #293c30 !important;
    -webkit-text-fill-color: #293c30 !important;
    background-color: #fffef9;
}

/* Labels and helper text should not inherit white theme text. */
[data-testid="stWidgetLabel"] *,
[data-testid="stHelp"] *,
[data-testid="stCaptionContainer"] * {
    color: #536454 !important;
}

/* Preserve white text only on the dark-green action buttons. */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Read-only weekend status box (matches the input styling). */
.status {
    background: #fffef9;
    border: 1px solid #d7dbd0;
    border-radius: 2px;
    padding: 0.55rem 0.75rem;
    min-height: 40px;
    color: #293c30;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# FILES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "best_model.pkl"
SCALER_FILE = BASE_DIR / "scaler.pkl"
ENCODER_FILE = BASE_DIR / "encoder.pkl"
NUMERIC_FILE = BASE_DIR / "numeric_features.pkl"
CATEGORICAL_FILE = BASE_DIR / "categorical_features.pkl"

required_files = [
    MODEL_FILE,
    SCALER_FILE,
    ENCODER_FILE,
    NUMERIC_FILE,
    CATEGORICAL_FILE
]

missing_files = [f.name for f in required_files if not f.exists()]

if missing_files:
    st.error(
        "Some required files are missing from the project folder: "
        + ", ".join(missing_files)
    )
    st.stop()


@st.cache_resource
def load_model_files():
    return (
        joblib.load(MODEL_FILE),
        joblib.load(SCALER_FILE),
        joblib.load(ENCODER_FILE),
        joblib.load(NUMERIC_FILE),
        joblib.load(CATEGORICAL_FILE)
    )


@st.cache_data
def load_csv_file(file_path):
    return pd.read_csv(file_path)


try:
    model, scaler, encoder, numeric_features, categorical_features = (
        load_model_files()
    )
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)
except Exception:
    st.error("The saved model files could not be loaded.")
    st.code(traceback.format_exc())
    st.stop()


csv_files = sorted(BASE_DIR.glob("*.csv"))

if not csv_files:
    st.error("Place your original dataset CSV in the same folder as app.py.")
    st.stop()

if len(csv_files) == 1:
    dataset_path = csv_files[0]
else:
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        [f.name for f in csv_files]
    )
    dataset_path = BASE_DIR / dataset_name

try:
    df = load_csv_file(str(dataset_path))
except Exception:
    st.error("The dataset could not be opened.")
    st.code(traceback.format_exc())
    st.stop()

DATASET_KEY = dataset_path.stem.replace(" ", "_")


# --------------------------------------------------
# VALIDATE DATASET AND FEATURE LISTS
# --------------------------------------------------

# The exact order used during training: numeric columns first, then
# categorical columns (this is how the processed row is assembled below).
expected_features = numeric_features + categorical_features

removed_columns = {"month", "collection_frequency_per_week"}
stale_removed_features = sorted(removed_columns.intersection(expected_features))
if stale_removed_features:
    st.error(
        "Your saved model artifacts still expect removed feature(s): "
        + ", ".join(stale_removed_features)
        + ". Rerun Colab training after removing these columns, then replace "
        "best_model.pkl, scaler.pkl, encoder.pkl, numeric_features.pkl and "
        "categorical_features.pkl with the newly generated files."
    )
    st.stop()

missing_columns = [c for c in expected_features if c not in df.columns]

if missing_columns:
    st.error(
        "These model input columns are missing from the dataset: "
        + ", ".join(missing_columns)
    )
    st.stop()

if df.empty:
    st.error("The dataset contains no records.")
    st.stop()

if set(numeric_features).intersection(categorical_features):
    st.error("A feature appears in both preprocessing feature lists.")
    st.stop()

for needed in ("city", "ward", "day_of_week", "weekend_flag"):
    if needed not in expected_features:
        st.error(
            f"'{needed}' is not in the saved feature lists, but this "
            "application needs it."
        )
        st.stop()


# --------------------------------------------------
# SIDEBAR AND INTRODUCTION
# --------------------------------------------------

# --------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div class='brand'>A city environment journal</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2 style="font-size:36px; line-height:1.05; margin-bottom:12px;">
            Waste<br>Notes.
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="color:#718071; font-size:14px; line-height:1.8;">
        A simple way to explore the relationship between city
        conditions and waste generation.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    st.markdown("**In this project**")
    st.caption("Smart Waste Generation Prediction")
    st.caption("Random Forest Regression")

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    st.caption(f"Dataset · {len(df):,} records")

    if "city" in df.columns:
        st.caption(f"Cities · {df['city'].nunique()}")

    if "waste_type" in df.columns:
        st.caption(f"Categories · {df['waste_type'].nunique()}")


# --------------------------------------------------
# MAIN INTRODUCTION
# --------------------------------------------------

st.markdown(
    "<div class='eyebrow'>Waste Notes &nbsp; / &nbsp; City study</div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero">
        A closer look at<br>
        what our cities <em>discard.</em>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="intro">
        Waste generation changes with the people who live in a place,
        local activity, weather and the day of the week. Adjust the
        details below to estimate waste generation under those conditions.
    </div>
    """,
    unsafe_allow_html=True
)

st.caption("")

st.markdown("<div class='rule'></div>", unsafe_allow_html=True)


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

DAY_ORDER = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]
WEEKEND_DAYS = {"Saturday", "Sunday"}

# Units read from the column-name suffix, e.g. temperature_c -> (°C)
UNIT_SUFFIXES = {"_c": "°C", "_mm": "mm", "_pct": "%"}

# Only the "+/-" button size. Any value can still be typed.
BUTTON_STEPS = {"population": 100, "households": 10}


def display_name(column):
    base = column
    for suffix in UNIT_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.replace("_", " ").strip().title()


def field_label(column):
    for suffix, unit in UNIT_SUFFIXES.items():
        if column.endswith(suffix):
            return f"{display_name(column)} ({unit})"
    return display_name(column)


def format_value(value):
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    return str(value)


def stretch(func, *args, **kwargs):
    """Full-width widgets across Streamlit versions."""
    try:
        return func(*args, width="stretch", **kwargs)
    except TypeError:
        return func(*args, use_container_width=True, **kwargs)


def encoder_categories():
    """Return {column: categories} from the fitted encoder."""
    categories = getattr(encoder, "categories_", None)
    if categories is None:
        return {}
    return dict(zip(categorical_features, [list(c) for c in categories]))


def most_common_index(options, column):
    """Index of the most frequent dataset value that is in options."""
    for value in df[column].dropna().value_counts().index:
        if value in options:
            return options.index(value)
    return 0


# --------------------------------------------------
# CITY -> WARD MAPPING (built from the dataset)
# --------------------------------------------------

known_wards = set(encoder_categories().get("ward", []))

city_ward_pairs = df[["city", "ward"]].dropna().drop_duplicates()

# A reliable mapping needs every ward to belong to exactly one city.
wards_in_many_cities = (
    city_ward_pairs.groupby("ward")["city"].nunique().loc[lambda s: s > 1]
)
if not wards_in_many_cities.empty:
    st.error(
        "The dataset does not contain a reliable city-to-ward mapping: "
        "these wards appear under more than one city: "
        + ", ".join(map(str, wards_in_many_cities.index[:10]))
    )
    st.stop()

city_to_wards = {}
for city_name, group in city_ward_pairs.groupby("city"):
    wards = sorted(w for w in group["ward"] if w in known_wards)
    if wards:
        city_to_wards[city_name] = wards

if not city_to_wards:
    st.error("No city/ward combinations could be built from the dataset.")
    st.stop()

city_options = sorted(city_to_wards)

day_options = [
    d for d in DAY_ORDER
    if d in encoder_categories().get("day_of_week", DAY_ORDER)
]
if not day_options:
    st.error("The encoder does not contain any weekday names.")
    st.stop()


# --------------------------------------------------
# INPUT WIDGETS
# --------------------------------------------------
# No min_value / max_value anywhere: the user may type any number.

input_values = {}   # exact values sent to the model
input_labels = {}   # friendlier text for the summary table
input_errors = []   # problems found while building the page


def add_city():
    city = st.selectbox(
        "City",
        city_options,
        index=most_common_index(city_options, "city"),
        key=f"city_{DATASET_KEY}",
    )
    input_values["city"] = city


def add_ward():
    # The ward list depends on the chosen city. Because the city dropdown is
    # NOT inside a form, changing it reruns the page immediately and this
    # list is rebuilt. The city is part of the widget key, so a ward chosen
    # for one city can never be carried over to another city.
    city = input_values["city"]
    ward_options = city_to_wards[city]
    ward = st.selectbox(
        "Ward / Zone",
        ward_options,
        key=f"ward_{DATASET_KEY}_{city}",
        help=f"Only wards that belong to {city} are listed.",
    )
    input_values["ward"] = ward


def add_day_of_week():
    day = st.selectbox(
        "Day Of Week",
        day_options,
        index=day_options.index("Wednesday") if "Wednesday" in day_options else 0,
        key=f"day_{DATASET_KEY}",
    )
    input_values["day_of_week"] = day


def add_weekend_flag():
    # Calculated from the day of week. The user cannot edit it.
    is_weekend = input_values["day_of_week"] in WEEKEND_DAYS
    input_values["weekend_flag"] = 1 if is_weekend else 0
    input_labels["weekend_flag"] = "Weekend (1)" if is_weekend else "Weekday (0)"

    st.markdown(
        f"""
        <div style="font-size:14px; margin-bottom:4px;">Weekend Status</div>
        <div class="status">{input_labels['weekend_flag']}</div>
        <div class="section-note" style="margin-top:6px;">
            Set automatically from the day of week.
        </div>
        """,
        unsafe_allow_html=True
    )


def add_festival_flag():
    choice = st.selectbox(
        "Festival",
        ["No Festival", "Festival"],
        key=f"festival_{DATASET_KEY}",
    )
    input_values["festival_flag"] = 1 if choice == "Festival" else 0
    input_labels["festival_flag"] = choice


def add_categorical(column):
    options = sorted(encoder_categories().get(column, []), key=str)
    if not options:
        input_errors.append(f"No categories were found for '{column}'.")
        return
    input_values[column] = st.selectbox(
        field_label(column),
        options,
        index=most_common_index(options, column),
        key=f"cat_{DATASET_KEY}_{column}",
    )


def add_number(column):
    values = (
        pd.to_numeric(df[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if values.empty:
        input_errors.append(f"'{column}' has no numeric values in the dataset.")
        return

    # Whole numbers in the dataset -> integer input, otherwise decimal input.
    is_whole = bool(np.allclose(values, values.round(), rtol=0, atol=1e-9))
    median = float(values.median())

    if is_whole:
        value = st.number_input(
            field_label(column),
            value=int(round(median)),
            step=int(BUTTON_STEPS.get(column, 1)),
            format="%d",
            key=f"num_{DATASET_KEY}_{column}",
        )
    else:
        value = st.number_input(
            field_label(column),
            value=float(round(median, 2)),
            step=0.1,
            format="%.2f",
            key=f"num_{DATASET_KEY}_{column}",
        )

    # A cleared box returns None; this is checked before predicting.
    input_values[column] = value


def add_input(column):
    if column == "city":
        add_city()
    elif column == "ward":
        add_ward()
    elif column == "day_of_week":
        add_day_of_week()
    elif column == "weekend_flag":
        add_weekend_flag()
    elif column == "festival_flag":
        add_festival_flag()
    elif column in categorical_features:
        add_categorical(column)
    elif column in numeric_features:
        add_number(column)


# --------------------------------------------------
# SCENARIO INPUTS
# --------------------------------------------------
# NOTE: there is deliberately no st.form here. Inside a form, widgets do not
# rerun the page until the submit button is pressed, so the ward list could
# not refresh when the city changes. Without a form every change reruns the
# page instantly and the button simply triggers the prediction.

st.markdown(
    "<div class='eyebrow'>01 &nbsp; / &nbsp; The scenario</div>",
    unsafe_allow_html=True
)

st.markdown("## Set the conditions")

st.markdown(
    """
    <div class="section-note">
        Start with a location and material type, then describe the
        population, weather and collection conditions.
    </div>
    """,
    unsafe_allow_html=True
)

# Order matters inside a group: city comes before ward, and day_of_week
# comes before weekend_flag, because the later one depends on the earlier.
group_definitions = [
    (
        "Location & waste",
        "Where the waste is generated and what kind of material it is.",
        ["city", "ward", "waste_type"]
    ),
    (
        "Population & activity",
        "People, households and activity that may affect local waste.",
        [
            "population",
            "households",
            "commercial_activity_index",
            "tourism_index",
            "festival_flag"
        ]
    ),
    (
        "Weather",
        "Weather conditions for the scenario.",
        ["temperature_c", "rainfall_mm", "humidity_pct"]
    ),
    (
        "Calendar",
        "Choose the day for your waste generation scenario.",
        [
            "day_of_week",
            "weekend_flag"
        ]
    )
]

used_features = set()

for heading, description, proposed_columns in group_definitions:
    available_columns = [c for c in proposed_columns if c in expected_features]

    if not available_columns:
        continue

    st.markdown(f"### {heading}")
    st.markdown(
        f"<div class='section-note'>{description}</div>",
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    for i, column in enumerate(available_columns):
        with (left if i % 2 == 0 else right):
            add_input(column)
        used_features.add(column)

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

# Any model feature not covered above still gets an input.
remaining_features = [c for c in expected_features if c not in used_features]

if remaining_features:
    st.markdown("### Other details")
    st.markdown(
        "<div class='section-note'>"
        "Additional variables used by the trained model."
        "</div>",
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    for i, column in enumerate(remaining_features):
        with (left if i % 2 == 0 else right):
            add_input(column)

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

submitted = stretch(
    st.button,
    "Estimate waste generation  →",
    key="estimate_button"
)


# --------------------------------------------------
# PRE-PREDICTION CHECKS
# --------------------------------------------------

def input_problems():
    """Return plain-language problems with the current inputs.
    Only checks that a usable value exists. It does NOT check ranges."""
    problems = list(input_errors)

    for column in expected_features:
        if column not in input_values:
            problems.append(f"No input was created for '{column}'.")

    for column in numeric_features:
        value = input_values.get(column)
        if value is None:
            problems.append(f"{field_label(column)}: please enter a number.")
        elif not np.isfinite(float(value)):
            problems.append(f"{field_label(column)}: the value is not finite.")

    # City / ward must belong together.
    city = input_values.get("city")
    ward = input_values.get("ward")
    if ward not in city_to_wards.get(city, []):
        problems.append(
            f"Ward '{ward}' does not belong to city '{city}'."
        )

    # Weekend flag must agree with the day of week.
    expected_weekend = 1 if input_values.get("day_of_week") in WEEKEND_DAYS else 0
    if input_values.get("weekend_flag") != expected_weekend:
        problems.append("The weekend flag does not match the day of week.")

    return problems


def preprocessing_problems():
    """Check the saved scaler/encoder against the feature lists."""
    problems = []

    scaler_names = getattr(scaler, "feature_names_in_", None)
    if scaler_names is not None and list(scaler_names) != numeric_features:
        problems.append(
            "The scaler was fitted on different numeric columns (or a "
            "different order) than numeric_features.pkl."
        )

    encoder_names = getattr(encoder, "feature_names_in_", None)
    if encoder_names is not None and list(encoder_names) != categorical_features:
        problems.append(
            "The encoder was fitted on different categorical columns (or a "
            "different order) than categorical_features.pkl."
        )

    return problems


def build_model_input():
    # Same order as training: scaled numeric columns, then encoded categories.
    numeric_input = pd.DataFrame(
        [[input_values[c] for c in numeric_features]],
        columns=numeric_features
    )
    categorical_input = pd.DataFrame(
        [[input_values[c] for c in categorical_features]],
        columns=categorical_features
    )

    # transform() only: the saved scaler and encoder are never refitted.
    scaled_numeric = scaler.transform(numeric_input)
    encoded_categorical = encoder.transform(categorical_input)

    if hasattr(encoded_categorical, "toarray"):
        encoded_categorical = encoded_categorical.toarray()

    processed = np.concatenate(
        [np.asarray(scaled_numeric), np.asarray(encoded_categorical)],
        axis=1
    )

    expected_width = getattr(model, "n_features_in_", None)
    if expected_width is not None and processed.shape[1] != expected_width:
        raise ValueError(
            f"The processed input has {processed.shape[1]} features, but the "
            f"model expects {expected_width}. The saved scaler, encoder and "
            "model do not match each other."
        )

    return processed


# --------------------------------------------------
# PREDICTION RESULT
# --------------------------------------------------

if submitted:
    problems = input_problems() + preprocessing_problems()

    if problems:
        st.session_state.pop("last_result", None)
        st.error("The estimate could not be generated:")
        for problem in problems:
            st.markdown(f"- {problem}")
    else:
        try:
            prediction = float(model.predict(build_model_input())[0])

            if not np.isfinite(prediction):
                raise ValueError("The model returned an invalid prediction.")

            summary = pd.DataFrame(
                [
                    {
                        "Condition": field_label(column),
                        "Value": input_labels.get(
                            column, format_value(input_values[column])
                        )
                    }
                    for column in expected_features
                ]
            )

            # Saved in session state so the result stays on screen when the
            # download button reruns the page.
            st.session_state["last_result"] = {
                "prediction": prediction,
                "summary": summary,
            }

        except Exception:
            st.session_state.pop("last_result", None)
            st.error("Something went wrong while generating the estimate.")
            st.code(traceback.format_exc())

result = st.session_state.get("last_result")

if result is not None:
    prediction = result["prediction"]
    summary = result["summary"]

    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='eyebrow'>02 &nbsp; / &nbsp; The estimate</div>",
        unsafe_allow_html=True
    )

    st.markdown("## Your result")

    st.markdown(
        f"""
        <div class="result">
            <div class="result-label">Estimated waste generated</div>
            <div class="result-number">
                {prediction:,.2f}
                <span style="font-size:19px;">tonnes/day</span>
            </div>
            <div class="result-caption">
                Random Forest Regression · Your selected scenario
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Your selected conditions")

    stretch(st.dataframe, summary, hide_index=True)

    st.caption(
        "This result belongs to the conditions listed above. "
        "Click the estimate button again after changing any input."
    )

    report = summary.copy()
    report.loc[len(report)] = {
        "Condition": "Estimated waste generated (tonnes/day)",
        "Value": format_value(round(prediction, 2))
    }

    st.download_button(
        "Download estimate",
        data=report.to_csv(index=False),
        file_name="waste_estimate.csv",
        mime="text/csv",
        key="download_estimate"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        WASTE NOTES &nbsp; · &nbsp; SMART WASTE GENERATION PREDICTION<br>
        
    </div>
    """,
    unsafe_allow_html=True
)