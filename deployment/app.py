import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px
import json
import random
import numpy as np

import sys
sys.path.append("..")
from src.trainer.moea_trainer import run_once, Args
from src.algorithms.multi_objective.MOEA_D_AM.prediction import predict_function


# Initialize session state variables
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'selected_labels' not in st.session_state:
    st.session_state.selected_labels = None
if 'cleaned_dataset' not in st.session_state:
    st.session_state.cleaned_dataset = None

if 'training_completed' not in st.session_state:
    st.session_state.training_completed = False
if 'all_results' not in st.session_state:
    st.session_state.all_results = pd.DataFrame()
if 'archive' not in st.session_state:
    st.session_state.archive = {}
if 'training_args' not in st.session_state:
    st.session_state.training_args = None
if 'cleaned_dataset' not in st.session_state:
    st.session_state.cleaned_dataset = None

st.set_page_config(page_title="MOEA/D-AM", page_icon="🧠", layout="wide")
# menu header
st.sidebar.title("MOEA/D-AM ALGORITHM")
# space
st.sidebar.markdown("")



def format_rule_from_antecedent(rule):
    """
    Convert rule antecedent format to readable text.
    Rule format: [(attr, val), (attr, val), ..., (class, prediction)]
    """
    if isinstance(rule, str):
        return rule  # Already formatted (e.g., "Majority Vote")
    
    conditions = []
    consequent = None
    
    for attr, val in rule[:-1]:  # All except last element are conditions
        if isinstance(val, str):
            conditions.append(f"({attr} == '{val}')")
        else:
            conditions.append(f"{attr} == {val})")
    
    # Last element is the consequent
    if len(rule) > 0:
        consequent_attr, consequent_val = rule[-1]
        consequent = f"({consequent_attr} = {consequent_val})"

    if conditions:
        rule_text = "IF " + " AND ".join(conditions)
        if consequent:
            rule_text += f" THEN {consequent}"
        return rule_text
    else:
        return f"Default rule: {consequent}" if consequent else "Default rule"
    

def make_predictions_with_rules(model, data_df, args):
    """
    Make predictions using your actual prediction functions and extract triggered rules.
    """
    
    # Set default parameters for prediction
    archive_type = args['archive_type']
    prediction_strat = args['prediction_strat']

    priors = model.get('priors', {})
    labels = list(priors.keys())
    task = 'single' if len(labels) == 1 else 'multi'

    archive = model.get('archive', {})

    
    # Use your updated prediction function
    if task == "multi":
        predictions_df, scores_df, triggered_rules = predict_function(
            data_df, archive, archive_type, prediction_strat, labels, priors, task
        )
        # Convert multi-label predictions to readable format
        predictions = []
        for _, row in predictions_df.iterrows():
            pred_labels = [val for i, val in enumerate(row)]
            predictions.append(pred_labels if pred_labels else ['no_label'])
        
        # Calculate average confidence across labels
        confidences = scores_df.mean(axis=1).tolist()
    else:
        predictions_series, scores, triggered_rules = predict_function(
            data_df, archive, archive_type, prediction_strat, labels, priors, task
        )
        predictions = predictions_series.tolist()
        confidences = triggered_rules['confidences']
    
    return predictions, triggered_rules
        
    


def get_readable_rules(archive, task, archive_structure):
    readable = []
    if task == "single":
        if archive_structure == "rules":
            for item in archive['run_1']['archive']:
                antecedent = " AND ".join([f"({term[0]} = {term[1]})" for term in item['rule'][:-1]])
                consequent = item['rule'][-1][1]
                readable.append(f"IF {antecedent} THEN (class = {consequent})")
        else:
            for item in archive['run_1']['archive']:
                ruleset = []
                for rule in item['ruleset']['rules']:
                    antecedent = " AND ".join([f"({term[0]} = {term[1]})" for term in rule['rule'][:-1]])
                    consequent = rule['rule'][-1][1]
                    ruleset.append(f"IF {antecedent} THEN (class = {consequent})")
                readable.append(ruleset)
    else:
        for item in archive['run_1']['archive']:
            ruleset = []
            for rule in item['ruleset']['rules']:
                antecedent = " AND ".join([f"({term[0]} = {term[1]})" for term in rule['rule'] if not term[0].startswith("label")])
                consequents = ", ".join([f"({term[0]} = {term[1]})" for term in rule['rule'] if term[0].startswith("label")])
                ruleset.append(f"IF {antecedent} THEN {consequents}")
            readable.append(ruleset)
    return readable

# --- Sidebar Layout ---
with st.sidebar:
    # Navigation menu (no title, no background box)
    selected = option_menu(
        menu_title=None,
        options=["Home", "Dataset & Preprocessing", "Model Training", "Prediction"],
        icons=["house", "table", "cpu", "magic"],
        default_index=0,
        orientation="vertical",
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "transparent",
            },
            "icon": {
                "color": "white",
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "18px",
                "text-align": "left",
                "margin": "4px 0",
                "color": "white",
                "border-radius": "6px",
                "padding": "8px 12px",
                # bold font
                "font-weight": "400",
                "--hover-color": "#334155",  # teal hover
            },
            "nav-link-selected": {
                "background-color": "#7345DF",  # indigo selected
                "color": "white",
            },
        },
        )

    st.markdown("---")
    st.sidebar.header("⚙️ Training Hyperparameters")

    #task = st.sidebar.selectbox("Task Type", ["single", "multi"], index=1)
    
    st.sidebar.subheader("Algorithm Control Parameters")
    options = {
        "Weighted Sum": "weighted",
        "Tchebycheff": "tchebycheff"
    }
    decomposition = st.sidebar.radio("Decomposition Approach", list(options.keys()), horizontal=True, index=0)
    decomposition = options[decomposition]

    population = st.sidebar.number_input("Population Size", 10, 1000, 50, 10)
    neighbors = st.sidebar.number_input("Neighborhood Size", 1, 50, 10, 1)
    groups = st.sidebar.number_input("Ant Groups", 1, 20, 5, 1)
    min_examples = st.sidebar.number_input("Min Samples per Rule", 1, 50, 10, 1)
    max_uncovered = st.sidebar.number_input("Max Uncovered", 1, 100, 10, 1)
    max_iter = st.sidebar.number_input("Max Iterations", 10, 1000, 100, 10)
    gamma = st.sidebar.slider("Gamma (Exploration/exploitation)", 0.0, 1.0, 0.9, 0.01)
    pruning = st.sidebar.checkbox("Enable Rule Pruning", value=False)

    st.markdown("---")
    
    st.sidebar.subheader("ACO-specific Parameters")
    alpha = st.sidebar.number_input("Alpha (pheromone influence)", 1, 10, 1)
    beta = st.sidebar.number_input("Beta (heuristic influence)", 1, 10, 1)
    p = st.sidebar.slider("P (evaporation rate)", 0.0, 1.0, 0.9, 0.01)
    epsilon = st.sidebar.slider("Epsilon (pheromones bounding factor)", 0.0, 1.0, 0.05, 0.01)

    st.markdown("---")

    st.sidebar.subheader("Archive-specific Parameters")
    archive_type = st.sidebar.radio("Archive Structure", ["rules", "rulesets"], horizontal=True, index=0)
    rulesets = st.sidebar.selectbox("Ruleset Formation", [None, "iteration", "subproblem"], index=0)
    rulesets_size = st.sidebar.number_input("Ruleset Size", 1, 10, 2)

    st.markdown("---")

    st.sidebar.subheader("Training-specific Parameters")
    prediction_strat = st.sidebar.radio("Prediction Strategy", ["all", "best", "reference"], index=0)
    cross_val = st.sidebar.checkbox("Cross Validation", value=False)
    folds = st.sidebar.number_input("Folds (if CV)", 3, 10, 5)
    random_state = st.sidebar.number_input("Random State", 0, 9999, None)
    runs = st.sidebar.number_input("Independent Runs", 1, 50, 1)

    # Store args in session state
    st.session_state.training_args = {
        "population": population,
        "neighbors": neighbors,
        "groups": groups,
        "min_examples": min_examples,
        "max_uncovered": max_uncovered,
        "max_iter": max_iter,
        "gamma": gamma,
        "alpha": alpha,
        "beta": beta,
        "p": p,
        "epsilon": epsilon,
        "pruning": pruning,
        "decomposition": decomposition,
        "archive_type": archive_type,
        "rulesets": rulesets,
        "rulesets_size": rulesets_size,
        "prediction_strat": prediction_strat,
        "cross_val": cross_val,
        "folds": folds,
        "random_state": None,
        "runs": runs
    }

# --- Home ---
if selected == "Home":
    st.title("🧠 MOEA/D-AM: Multi-Objective Evolutionary Algorithm based on Decomposition and Ant-Miner")
    st.write("""
    Welcome to the MOEA/D-AM application! This tool allows you to preprocess datasets and train the MOEA/D-AM algorithm for classification tasks.
    
    **Navigation:**
    - Use the sidebar to navigate between sections.
    - Start by uploading and preprocessing your dataset in the "Dataset & Preprocessing" section.
    - Once your data is ready, proceed to "Model Training" to configure hyperparameters and run the training process.
    
    **About MOEA/D-AM:**
    MOEA/D-AM is a multi-objective evolutionary algorithm that combines the principles of Ant Colony Optimization with decomposition strategies. It is designed to generate interpretable classification rules while optimizing multiple objectives such as accuracy and rule simplicity.
    
    **Getting Started:**
    1. Upload your dataset in CSV format.
    2. Select label columns and visualize class distributions.
    3. Preprocess the data to handle missing values, duplicates, and discretization.
    4. Configure training hyperparameters and run the MOEA/D-AM algorithm.
    
    Enjoy exploring and training with MOEA/D-AM!
    """)

# --- Dataset & Preprocessing ---
elif selected == "Dataset & Preprocessing":

    st.header("📊 Dataset and Preprocessing")
    st.write("""Upload your dataset in CSV format. You can preview the data and check basic statistics.""")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file:
        # Read dataset
        df = pd.read_csv(uploaded_file)
        st.session_state["dataset"] = df

        # Dataset Information
        st.subheader("🔍 Dataset Information")
        st.dataframe(df.head(), width="stretch")
    
        # Basic Info in 2 columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Rows", df.shape[0])
        with col2:
            st.metric("Number of Columns", df.shape[1])
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("Number of Missing Values", missing)
    
        # Summary Statistics
        with st.expander("📌 Show Summary Statistics"):
            st.write(df.describe(include="all").transpose())
    
        st.markdown("---")
    
        # --- Label Selection ---
        st.subheader("Select Label(s)")
        label_cols = st.multiselect(
            "Select Label Columns",
            options=df.columns.tolist(),
            default=None,
            placeholder="Type to search columns...",
            label_visibility="visible"
            )
    
        if label_cols:
            st.success(f"Selected labels: {', '.join(label_cols)}")
    
            # --- Class Distribution Visualization ---
            st.subheader("Class Distribution Across Labels")
    
            # Reshape → long format
            melted = df.melt(value_vars=label_cols, var_name="Label", value_name="Class")
    
            # Count occurrences
            counts = melted.groupby(["Label", "Class"]).size().reset_index(name="Count")
    
            # Plot
            fig = px.bar(
                counts,
                x="Label",
                y="Count",
                color="Class",
                barmode="group",
                text="Count",
                title="Class Distribution Across Labels"
            )
    
            fig.update_layout(
                xaxis_title="Label",
                yaxis_title="Count",
                legend_title="Class Value",
                bargap=0.2
            )

            n_labels = len(label_cols)

            if n_labels > 0:
                bar_width = min(0.8, 0.25 + (n_labels - 1) * 0.05)
            else:
                bar_width = 0.25 
            
            fig.update_traces(width=bar_width)
    
            st.plotly_chart(fig, width=True)

            st.session_state["selected_labels"] = label_cols

        else:
            st.warning("Please select at least one label column to display class distribution.")

        
        # ------------------ PREPROCESSING ------------------
        st.markdown("---")
        if st.session_state.selected_labels:
            st.subheader(" Data Preprocessing")
        
            if st.button("Run Preprocessing"):
                with st.spinner("Preprocessing dataset..."):
                    data = df.copy()
        
                    # --- Handle label renaming ---
                    label_mapping = {}
                    if len(st.session_state.selected_labels) == 1:
                        # Single-label case
                        original = st.session_state.selected_labels[0]
                        data = data.rename(columns={original: "class"})
                        label_mapping[original] = "class"
                        new_label_cols = ["class"]
                    else:
                        # Multi-label case
                        if 'label' not in st.session_state.selected_labels[0]:
                            for i, col in enumerate(st.session_state.selected_labels):
                                new_name = f"label{i+1}"
                                data = data.rename(columns={col: new_name})
                                label_mapping[col] = new_name
                            new_label_cols = list(label_mapping.values())
                        else:
                            new_label_cols = st.session_state.selected_labels
        
                    # --- Fill missing values ---
                    
                    data = data.fillna("unknown")
        
                    # --- Drop duplicates ---
                    #data = data.drop_duplicates().reset_index(drop=True)
        
                    # --- Drop constant columns ---
                    data = data.loc[:, data.nunique() != 1]
        
                    # --- Discretization for high-cardinality columns ---
                    for col in data.columns:
                        if col not in new_label_cols and data[col].dtype in [int, float]:
                            if data[col].nunique() > 10:
                                st.write(f'Discretizing column "{col}" with {data[col].nunique()} unique values into 5 bins.')
                                data[col] = pd.qcut(data[col], q=5, labels=False, duplicates='drop')

                    # --- Type casting ---
                    if len(new_label_cols) == 1:
                        # Single-label → everything as string
                        for col in data.columns:
                            data[col] = data[col].astype(str)
                    else:
                        # Multi-label → everything as int
                        for col in data.columns:
                            if col not in new_label_cols:
                                if data[col].dtype == object:
                                    data[col] = data[col].astype("category").cat.codes
        
                    # --- Save preprocessed dataset & mapping in session ---
                    st.session_state["preprocessed_data"] = data
                    st.session_state["label_mapping"] = label_mapping
                    st.session_state["task"] = "single" if len(new_label_cols) == 1 else "multi"
        
                    # --- Show results ---
                    st.success("✅ Preprocessing completed successfully!")
                    st.write("#### Label Mapping")
                    st.json(label_mapping)
        
                    st.write("#### Preview of Preprocessed Data")
                    st.dataframe(data.head(), width="stretch")

                    # --- Download cleaned dataset ---
                    csv = data.to_csv(index=False).encode("utf-8")
                    st.session_state["cleaned_dataset"] = csv

                    dataset_name = uploaded_file.name if uploaded_file else "dataset.csv"

                    st.download_button(
                        label="📥 Download Cleaned Dataset",
                        data=st.session_state.cleaned_dataset,
                        file_name=dataset_name.replace(".csv", "_cleaned.csv"),
                        mime="text/csv",
                        key="download_cleaned_dataset"
                    )
        
    else:
        st.warning("📂 Upload a CSV file to get started.")

elif selected == "Model Training":
    st.header("🧪 Model Training with MOEA/D-AM")

    if "preprocessed_data" not in st.session_state:
        st.warning("⚠️ Please import and preprocess your dataset first in 'Dataset & Preprocessing'.")
    else:
        data = st.session_state["preprocessed_data"]
        label_mapping = st.session_state.get("label_mapping", {})
        task = st.session_state.get("task", "single")

        # --- Prepare Data ---
        data = data.astype(str)

        labels = ["class"] if task == "single" else [col for col in data.columns if "label" in col]
        X = data.drop(columns=labels)
        y = data[labels] if task == "multi" else data["class"]

        all_results = pd.DataFrame()
        archive = {}

        # --- Train Button ---
        if st.button("🚀 Run Training"):
            with st.spinner("Running MOEA/D-AM... this may take a while ⏳"):
        
                # --- Prepare arguments ---
                args = Args(
                    task=task, population=population, neighbors=neighbors, groups=groups,
                    min_examples=min_examples, max_uncovered=max_uncovered, max_iter=max_iter,
                    gamma=gamma, alpha=alpha, beta=beta, p=p, pruning=int(pruning),
                    decomposition=decomposition, archive_type=archive_type,
                    rulesets=rulesets, rulesets_size=rulesets_size,
                    prediction_strat=prediction_strat,
                    cross_val=cross_val, folds=folds, random_state=random_state, runs=runs
                )
        
                all_results = pd.DataFrame()
                archive = {}
        
                # --- Training ---
                if runs > 1:
                    for run_id in range(1, runs + 1):
                        results, archive = run_once(args, X, y, labels, run_id=run_id, archive=archive)
                        all_results = pd.concat([all_results, results], ignore_index=True)
                    st.success(f"✅ All {runs} Runs Completed!")
                else:
                    results, archive = run_once(args, X, y, labels, run_id=1, archive=archive)
                    all_results = pd.concat([all_results, results], ignore_index=True)
                    st.success("✅ Training Completed!")
        
                # --- SECTION 1: TRAINING RESULTS ---
                st.subheader("📊 Training Results")
                
                # Enhance results table
                display_df = all_results.copy()
                average = pd.DataFrame(columns=[col for col in display_df.columns if col not in ['run', 'fold']])
                if runs == 1 and cross_val:
                    display_df = display_df.drop(columns=['run'])
                    average = pd.DataFrame(display_df.mean(axis=0)).T
                    average.index = ["Average"]
                elif runs > 1:
                    display_df = pd.DataFrame(display_df.mean(axis=0)).T
                    display_df.index = ["Average"]
        
                # Display results table
                st.dataframe(display_df.style.format("{:.4f}"), width="stretch")
                if runs == 1 and cross_val:
                    st.dataframe(average.style.format("{:.4f}"), width="stretch")
                
                # Results download button
                csv = all_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="moead_am_results.csv",
                    mime="text/csv",
                    width="stretch"
                )
        
                # --- SECTION 2: LEARNED RULES ---
                st.markdown("---")
                st.subheader("🎯 Learned Rules")
                
                # Display readable rules only for single-run, no CV
                if runs == 1 and not cross_val:
                    readable_rules = get_readable_rules(archive, task, archive_structure=archive_type)
                    
                    if readable_rules:
                        # Format rules for better display
                        formatted_rules = []
                        for i, ruleset in enumerate(readable_rules, start=1):
                            if isinstance(ruleset, list):
                                rules_text = "\n".join(ruleset)
                            else:
                                rules_text = ruleset
                            
                            formatted_rules.append(f"// Ruleset {i}\n{rules_text}")
                        
                        all_rules_text = "\n\n".join(formatted_rules)
                        
                        # Display in code block style
                        st.code(all_rules_text, height=250, language="typescript")

                        # Model download button (JSON archive)
                        import json
                        archive_json = json.dumps(archive, indent=2, default=str).encode("utf-8")
                        st.download_button(
                            label="📦 Download Model (JSON)",
                            data=archive_json,
                            file_name="moead_am_model.json",
                            mime="application/json",
                            width="stretch"
                        )
                    else:
                        st.info("No rules to display for current configuration.")
                else:
                    st.info("📝 Rules display is only available for single-run without cross-validation.")
                    
                    # Still provide model download for other configurations
                    import json
                    archive_json = json.dumps(archive, indent=2, default=str).encode("utf-8")
                    st.download_button(
                        label="📦 Download Model (JSON)",
                        data=archive_json,
                        file_name="moead_am_model.json",
                        mime="application/json",
                        width="stretch"
                    )
        
                # --- SECTION 3: VISUALIZATION ---
                st.markdown("---")
                st.subheader("📈 Evolution & Pareto Front")
                col1, col2 = st.columns(2)
        
                # --- Hypervolume Evolution ---
                with col1:
                    st.markdown("**Hypervolume Evolution**")
                    import matplotlib.pyplot as plt
                    import numpy as np
        
                    hv_histories = []
                    for run_id in range(1, runs + 1):
                        run_hv_history = []
                        if cross_val:
                            for fold_id in range(1, folds + 1):
                                run_hv_history.append(archive[f"run_{run_id}"][f"fold_{fold_id}"]['hv'])
                        else:
                            run_hv_history.append(archive[f"run_{run_id}"]['hv'])

                        min_length = min(len(hv) for hv in run_hv_history)
                        run_hv_history = [hv[:min_length] for hv in run_hv_history]
                        avg_run_hv_history = np.mean(run_hv_history, axis=0)
                        hv_histories.append(avg_run_hv_history)
        
                    min_length = min(len(hv) for hv in hv_histories)
                    hv_histories = [hv[:min_length] for hv in hv_histories]
                    avg_hv_history = np.mean(hv_histories, axis=0)
                    std_hv_history = np.std(hv_histories, axis=0)
        
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.set_title("Hypervolume Evolution", fontsize=12, weight='bold')
                    ax.plot(avg_hv_history, color='black', linewidth=2, marker='o', markersize=4,
                            markerfacecolor='black', label='Avg Hypervolume')
                    ax.fill_between(range(len(avg_hv_history)),
                                    avg_hv_history - std_hv_history,
                                    avg_hv_history + std_hv_history,
                                    color='gray', alpha=0.3, label='± Std. Dev')
                    ax.set_xlabel("Generations", fontsize=12)
                    ax.set_ylabel("Hypervolume (HV)", fontsize=12)
                    ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
                    ax.legend(frameon=True, fontsize=10, loc="lower right")
                    st.pyplot(fig)
        
                # --- Pareto Front ---
                with col2:
                    st.markdown("**Pareto Front**")
                    all_solutions = []
                    local_pareto = []
        
                    for run_id in range(1, runs + 1):
                        if cross_val:
                            for fold_id in range(1, folds + 1):
                                fold_data = archive[f"run_{run_id}"][f"fold_{fold_id}"]
                                all_solutions.extend(fold_data["all"])
                                if task == 'single':
                                    for sol in fold_data["archive"]:
                                        local_pareto.append(sol["fitness"])
                                else:
                                    for sol in fold_data["archive"]:
                                        local_pareto.append(sol["ruleset"]["fitness"])
                        else:
                            fold_data = archive[f"run_{run_id}"]
                            all_solutions.extend(fold_data["all"])
                            if task == 'single':
                                for sol in fold_data["archive"]:
                                    local_pareto.append(sol["fitness"])
                            else:
                                for sol in fold_data["archive"]:
                                    local_pareto.append(sol["ruleset"]["fitness"])
        
                    all_solutions = np.array(all_solutions, dtype=float).reshape(-1, 2)
                    local_pareto = np.array(local_pareto, dtype=float).reshape(-1, 2)
        
                    def is_dominated(point, archive):
                        return any((a[0] >= point[0] and a[1] >= point[1]) for a in archive)
        
                    non_dominated_all = np.array([p for p in all_solutions if not is_dominated(p, local_pareto)])
                    dominated_all = np.array([p for p in all_solutions if is_dominated(p, local_pareto)])
        
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.set_title("Pareto Front", fontsize=12, weight='bold')
        
                    if dominated_all.size > 0:
                        ax.scatter(dominated_all[:, 1], dominated_all[:, 0],
                                   color='lightgrey', alpha=0.6, s=20, label="Dominated by Archive")
                    if non_dominated_all.size > 0:
                        ax.scatter(non_dominated_all[:, 1], non_dominated_all[:, 0],
                                   color='grey', alpha=0.6, s=25, label="Non-dominated (wrt Archive)")
                    if local_pareto.size > 0:
                        ax.scatter(local_pareto[:, 1], local_pareto[:, 0],
                                   color='red', edgecolor='black', s=40, label="Local Pareto (Archive)")
        
                    ax.set_xlabel("Simplicity", fontsize=12)
                    ax.set_ylabel("Confidence", fontsize=12)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    ax.legend(frameon=True, fontsize=10)
                    st.pyplot(fig)

        # Save training state
        st.session_state.training_completed = True
        st.session_state.all_results = all_results
        st.session_state.archive = archive
        st.session_state.training_config = {'task': task, 'runs': runs, 'cross_val': cross_val}
                

# Add this to your main sidebar selection logic:
# selected = st.sidebar.selectbox("Choose a page", ["Training", "Prediction", ...])

elif selected == "Prediction":
    st.title("🔮 Model Prediction")
    st.markdown("Make predictions using trained MOEA/D-AM models")

    # Initialize session state for prediction page
    if 'selected_archive' not in st.session_state:
        st.session_state.selected_archive = None
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False
    if 'model_config' not in st.session_state:
        st.session_state.model_config = {}


    # Step 1: Model Selection
    st.subheader("📦 Select Model")
    
    model_choice = st.radio(
        "Choose your model source:",
        ["Use trained model from Training page", "Upload existing model (JSON)"],
        key="model_choice_pred",
        horizontal=True
    )

    if model_choice == "Use trained model from Training page":
        # Check if there's a trained model from the training page
        if hasattr(st.session_state, 'training_completed') and st.session_state.training_completed and st.session_state.archive:
            st.success("✅ Using model from Training page session")
            st.session_state.selected_archive = st.session_state.archive
            st.session_state.model_config = st.session_state.training_config

            nb_rules = 0
            if st.session_state.model_config['task'] == 'single':
                nb_rules = len(st.session_state.selected_archive.get('run_1', {}).get('archive', []))
            else:
                ruleset_length = 0
                for ruleset in st.session_state.selected_archive.get('run_1', {}).get('archive', []):
                    ruleset_length += len(ruleset.get('ruleset', {}).get('rules', []))
                nb_rules = ruleset_length
            
            # Show model info
            with st.expander("📋 Model Information"):
                st.write(f"**Task**: {st.session_state.model_config.get('task', 'Unknown')}")
                st.write(f"**Number of Rules**: {nb_rules}")
                st.write(f"**File Size**: {len(str(st.session_state.selected_archive))} characters")
                
        else:
            st.warning("⚠️ No trained model found from Training page. Please train a model first or upload one.")
            st.session_state.selected_archive = None
    
    else:  # Upload model
        uploaded_model = st.file_uploader(
            "Upload your model (JSON file)", 
            type=['json'],
            key="model_upload_pred",
            help="Upload a JSON file exported from a previous training session"
        )
        
        if uploaded_model is not None:
            try:
                import json
                model_content = json.loads(uploaded_model.read())
                st.session_state.selected_archive = model_content
                
                # Try to extract or set default config
                st.session_state.model_config = {
                    # task single if 'class' in model_content[run][priors]
                    'task': 'single' if 'class' in model_content.get('run_1', {}).get('priors', {}) else 'multi',
                    'runs': len(model_content) if isinstance(model_content, dict) else 1,
                    'cross_val': False,
                }
                
                st.success("✅ Model uploaded successfully!")

                nb_rules = 0
                if st.session_state.model_config['task'] == 'single':
                    nb_rules = len(model_content.get('run_1', {}).get('archive', []))
                else:
                    ruleset_length = 0
                    for ruleset in model_content.get('run_1', {}).get('archive', []):
                        ruleset_length += len(ruleset.get('ruleset', {}).get('rules', []))
                    nb_rules = ruleset_length
                
                # Show model info
                with st.expander("📋 Uploaded Model Information"):
                    st.write(f"**Task**: {st.session_state.model_config['task']}")
                    st.write(f"**Number of Rules**: {nb_rules}")
                    st.write(f"**File Size**: {len(str(model_content))} characters")
                
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.session_state.selected_archive = None
        else:
            st.session_state.selected_archive = None


    # Step 2: Upload Prediction Data (only if model is selected)
    if st.session_state.selected_archive is not None:
        st.markdown("---")
        st.subheader("📄 Upload Data for Prediction")

        st.session_state.task = 'single' if 'class' in st.session_state.selected_archive.get('run_1', {}).get('priors', {}) else 'multi'

        uploaded_data = st.file_uploader(
            "Upload CSV file with samples to predict", 
            type=['csv'],
            key="data_upload_pred",
            help="Upload a CSV file containing the features for prediction. Make sure column names match your training data."
        )
        
        if uploaded_data is not None:
            try:
                prediction_df = pd.read_csv(uploaded_data, dtype=str)
                st.session_state.prediction_data = prediction_df
                
                st.success(f"✅ Data uploaded successfully! **{len(prediction_df)} samples** loaded.")
                
                # Display data info and preview
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📊 Samples", len(prediction_df))
                with col2:
                    st.metric("📋 Features", len(prediction_df.columns))
                
                # Data preview
                with st.expander("📊 Data Preview"):
                    st.dataframe(prediction_df.head(10), width="stretch")
                
                # Step 3: Make Predictions
                st.markdown("---")
                st.subheader("🎯 Make Predictions")
                
                if st.button("🚀 Generate Predictions", type="primary"):
                    with st.spinner("Making predictions... Please wait"):
                        try:
                            archive = st.session_state.selected_archive.get('run_1', {})
                                                        
                            # Call prediction function
                            predictions, triggered_rules = make_predictions_with_rules(
                                archive, 
                                prediction_df,
                                st.session_state.training_args
                            )
                            
                            
                            # Store predictions in session state
                            st.session_state.predictions = predictions
                            st.session_state.triggered_rules = triggered_rules
                            st.session_state.predictions_made = True
                            
                            st.success("✅ Predictions completed successfully!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                            st.write("Please check that your data format matches the training data.")

            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.info("💡 Make sure your CSV file is properly formatted with headers.")
                st.session_state.prediction_data = None 
    
    # Step 4: Display Results (if predictions were made)
    if st.session_state.predictions_made and hasattr(st.session_state, 'predictions'):
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Create results DataFrame
        results_df = st.session_state.prediction_data.copy()
        results_df['Prediction'] = st.session_state.predictions
        results_df['Confidence'] = [conf for conf in st.session_state.triggered_rules['confidences']]
        
        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Samples", len(results_df))
        with col2:
            avg_confidence = sum(st.session_state.triggered_rules['confidences']) / len(st.session_state.triggered_rules['confidences'])
            st.metric("📈 Avg Confidence", f"{avg_confidence:.3f}")

        if st.session_state.task == 'single':
            with col3:
                unique_predictions = len(set(st.session_state.predictions))
                st.metric("🏷️ Unique Classes", unique_predictions)
        
        # Display results table
        st.dataframe(results_df, width="stretch")
        
        # Download predictions
        csv_predictions = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions CSV",
            data=csv_predictions,
            file_name="predictions.csv",
            mime="text/csv",
            width="stretch",
            key=f"download_predictions_{hash(str(results_df.values.tobytes()))}"
        )
        
        # Step 5: Rule Explanation
        st.markdown("---")
        st.subheader("🔍 Rule Explanation & Analysis")
        st.write("Select a sample to see the triggered rule and understand the prediction:")
        
        # Row selection with better formatting
        selected_row = st.selectbox(
            "Choose a sample to analyze:",
            options=range(len(results_df)),
            format_func=lambda x: f"Sample {x}: Prediction = {results_df.iloc[x]['Prediction']}",
            key="row_selector_pred",
        )
        
        if selected_row is not None:
            # Display selected sample analysis
            st.markdown(f"### 📋 Analysis for Sample {selected_row + 1}")
            
            # Create three columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔹 Input Features:**")
                sample_data = st.session_state.prediction_data.iloc[selected_row].to_dict()
                
                # Display features in a nice format
                feature_df = pd.DataFrame(list(sample_data.items()), columns=['Feature', 'Value'])
                st.dataframe(feature_df, width="stretch", hide_index=True)

            with col2:
                st.markdown("**🔹 Prediction Result:**")
                pred_info = {
                    'Sample Index': selected_row,
                    'Predicted Class': st.session_state.predictions[selected_row],
                    'Confidence Score': f"{st.session_state.triggered_rules['confidences'][selected_row]:.3f}",                    
                }
                
                for key, value in pred_info.items():
                    st.write(f"• **{key}**: {value}")

                # space
                st.write("")

                # Confidence gauge (visual indicator)
                confidence_val = st.session_state.triggered_rules['confidences'][selected_row]
                if confidence_val >= 0.75:
                    conf_color = "🟢 High"
                elif confidence_val >= 0.5:
                    conf_color = "🟡 Medium"
                else:
                    conf_color = "🔴 Low"
                
                st.markdown("**🔹 Confidence Level:**")
                st.write(conf_color)
                st.progress(confidence_val)
                
            
            # Display triggered rule
            st.markdown("### ⚡ Triggered Rule")
            triggered_rule = st.session_state.triggered_rules['rules'][selected_row]

            if st.session_state.task == 'multi':
                # For multi-label, triggered_rule is a ruleset
                st.write("This is a ruleset for multi-label classification. Displaying all rules in the set:")
                all_formatted_rule = ""
                for i, rule in enumerate(triggered_rule, start=1):
                    formatted_rule = format_rule_from_antecedent(rule)
                    all_formatted_rule += f"{formatted_rule}\n"
                st.code(all_formatted_rule, language="typescript")
            else:
                # For single-label, triggered_rule is a single rule
                formatted_rule = format_rule_from_antecedent(triggered_rule)
                st.code(formatted_rule, language="typescript")

    # Action buttons at the bottom
    if st.session_state.predictions_made:
        st.markdown("---")
        
        if st.button("🔄 Make New Predictions", width='stretch'):
            # Reset prediction state but keep the model
            st.session_state.prediction_data = None
            st.session_state.predictions_made = False
            if hasattr(st.session_state, 'predictions'):
                delattr(st.session_state, 'predictions')
                delattr(st.session_state, 'triggered_rules')
            st.rerun()


    
                        
                        
           
    
    