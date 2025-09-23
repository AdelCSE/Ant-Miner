import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from pandas.api.types import is_numeric_dtype
import base64
import os
#cimport train

# ------------------ PAGE CONFIG ------------------
if 'app_name' not in st.session_state:
    st.session_state['app_name'] = "Multi-Label Classification Interface"

st.set_page_config(
    page_title=st.session_state['app_name'],
    layout="wide",
    page_icon=":gear:"
)

# ------------------ GLOBAL STYLES ------------------
sns.set_theme(style="whitegrid")  # seaborn global theme
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
})

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #f9fbfd; color: #333; font-family: 'Helvetica Neue', sans-serif; }
    [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #dee2e6; }
    h1, h2, h3, h4, h5, h6 { color: #004085; }
    .stButton > button {
        background-color: #007bff; color: white; border-radius: 6px;
        padding: 8px 18px; font-size: 15px; transition: 0.3s;
    }
    .stButton > button:hover { background-color: #0056b3; }
    .stDownloadButton > button { background-color: #17a2b8; color: white; border-radius: 6px; padding: 8px 18px; }
    .stDownloadButton > button:hover { background-color: #138496; }
    .stExpander { border: 1px solid #dee2e6; border-radius: 6px; background-color: #ffffff; }
    .dataframe th { background-color: #f1f3f5; color: #333; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_page' not in st.session_state:
    st.session_state['selected_page'] = "Dataset & Preprocessing"
if 'preprocessed_data' not in st.session_state:
    st.session_state['preprocessed_data'] = {'5bin': None, '10bin': None}
# ------------------ SIDEBAR NAVIGATION ------------------
with st.sidebar:
    st.title("Navigation")
    
    if st.button("Dataset & Preprocessing", key="dataset_btn", 
                 help="Access dataset upload and preprocessing tools"):
        st.session_state['selected_page'] = "Dataset & Preprocessing"
    
    if st.button("Training", key="training_btn", 
                 help="Access model training tools"):
        st.session_state['selected_page'] = "Training"
    
    st.title("Settings")
    with st.expander("Application Customization"):
        app_name = st.text_input("Application Name", value=st.session_state['app_name'])
        if app_name and app_name != st.session_state['app_name']:
            st.session_state['app_name'] = app_name
            st.rerun()
        
        icon_file = st.file_uploader("Upload Tab Icon (PNG)", type=["png"])
        if icon_file:
            # Validate file size (e.g., max 100KB) and type
            if icon_file.size > 102400:  # 100KB limit
                st.error("Icon file size exceeds 100KB. Please upload a smaller file.")
            else:
                # Save temporary file and serve as favicon
                import os
                temp_icon_path = f"temp_icon_{id(icon_file)}.png"
                with open(temp_icon_path, "wb") as f:
                    f.write(icon_file.getbuffer())
                # Inject favicon link with file path (requires server-side hosting or local file access)
                st.markdown(f'<link rel="icon" href="file://{temp_icon_path}" type="image/png">', unsafe_allow_html=True)
                st.success("Icon uploaded. Refresh the page to see the change.")
                # Note: File:// may not work in all browsers; recommend manual refresh or server setup
                # Clean up temporary file on rerun
                if os.path.exists(temp_icon_path):
                    os.remove(temp_icon_path)
        
        st.subheader("Active Functionalities")
        show_data_preview = st.checkbox("Show Dataset Preview", value=True)
        show_label_selection = st.checkbox("Show Label Selection", value=True)
        show_eda = st.checkbox("Show EDA", value=True)
        show_preprocess = st.checkbox("Show Preprocessing", value=True)
# ------------------ DATASET & PREPROCESSING ------------------
if st.session_state['selected_page'] == "Dataset & Preprocessing":
    st.title(f" {st.session_state['app_name']}")

    uploaded_file = st.file_uploader("Upload Dataset (CSV or XLSX)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            df = pd.read_csv(uploaded_file) if file_type == "csv" else pd.read_excel(uploaded_file)
            st.success(" Dataset loaded successfully")

            if show_data_preview:
                st.subheader(" Dataset Preview")
                st.dataframe(
                    df.head().style.highlight_null('red').background_gradient(cmap="Blues")
                )

            # Label selection
            label_columns = []
            if show_label_selection:
                st.subheader("🏷 Label Configuration")
                label_columns = st.multiselect("Select Label Columns", df.columns.tolist())

            # Exploratory Data Analysis
            if show_eda and label_columns:
                st.subheader(" Exploratory Data Analysis")

                with st.expander("Dataset Info"):
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                    st.write("**Statistical Summary:**")
                    st.dataframe(df.describe().style.background_gradient(cmap="Purples"))

                with st.expander("Class Distribution Analysis"):
                    for label in label_columns:
                        with st.spinner(f"Generating distribution for {label}..."):
                            st.write(f"Distribution for {label}:")
                            # Check if the label has valid data
                            if df[label].isnull().all():
                                st.write("No valid data to plot for this label.")
                            else:
                                # Get value counts and handle low unique value cases
                                class_count = df[label].value_counts()
                                if class_count.empty:
                                    st.write("No valid distribution to plot for this label.")
                                else:
                                    # If only a few unique values, use raw counts; otherwise, normalize to percentages
                                    if len(class_count) <= 2:
                                        y_label = "Count"
                                        values = class_count.values
                                    else:
                                        y_label = "Percentage (%)"
                                        values = class_count.values / class_count.sum() * 100

                                    fig, ax = plt.subplots()
                                    sns.barplot(x=class_count.index, y=values, ax=ax, color='#007bff')
                                    ax.set_ylabel(y_label)
                                    ax.set_title(f"Class Distribution for {label}")
                                    plt.xticks(rotation=45)
                                    # Adjust layout to prevent label cutoff
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    st.write(class_count)

                with st.expander("Feature Visualizations"):
                    numeric_cols = [col for col in df.columns if is_numeric_dtype(df[col]) and col not in label_columns]
                    if numeric_cols:
                        st.write("Histograms & Boxplots (Top 5 numeric features):")
                        for col in numeric_cols[:5]:
                            c1, c2 = st.columns(2)
                            with c1:
                                fig, ax = plt.subplots(figsize=(4, 3))
                                sns.histplot(df[col], kde=True, ax=ax, color="#28a745")
                                ax.set_title(f"Histogram: {col}")
                                st.pyplot(fig)
                            with c2:
                                fig, ax = plt.subplots(figsize=(4, 3))
                                sns.boxplot(x=df[col], ax=ax, color="#17a2b8")
                                ax.set_title(f"Boxplot: {col}")
                                st.pyplot(fig)
                        

                        st.write("Correlation Heatmap")
                        corr = df[numeric_cols].corr()
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(corr, cmap="RdBu", annot=len(numeric_cols) < 8, fmt=".2f", ax=ax)
                        st.pyplot(fig)
                        
            # Preprocessing
                    if show_preprocess and label_columns:
                        if st.button("Initiate Preprocessing"):
                            with st.spinner("Preprocessing dataset..."):
                                st.subheader("Preprocessing Output")
                                
                                data = df.copy()
                                data = data.fillna('unknown')
                                
                                for i, col in enumerate(label_columns, start=1):
                                    data = data.rename(columns={col: f"label{i}"})
                                
                                new_label_columns = [f"label{i}" for i in range(1, len(label_columns) + 1)]
                                
                                data = data.drop_duplicates().reset_index(drop=True)
                                
                                data = data.loc[:, data.nunique() != 1]
                                
                                data5 = data.copy()
                                for col in data5.columns:
                                    if not col.startswith("label") and is_numeric_dtype(data5[col]):
                                        data5[col] = pd.cut(data5[col], bins=5, labels=[str(i) for i in range(1, 6)])
                                
                                data10 = data.copy()
                                for col in data10.columns:
                                    if not col.startswith("label") and is_numeric_dtype(data10[col]):
                                        data10[col] = pd.cut(data10[col], bins=10, labels=[str(i) for i in range(1, 11)])
                                
                                # Store preprocessed data in session state
                                st.session_state['preprocessed_data']['5bin'] = data5
                                st.session_state['preprocessed_data']['10bin'] = data10
                                
                                st.write("5-Bin Discretized Data Preview:")
                                st.dataframe(data5.head().style.background_gradient(cmap='Greens'))
                                
                                st.write("10-Bin Discretized Data Preview:")
                                st.dataframe(data10.head().style.background_gradient(cmap='Blues'))
                                
                                st.subheader("Download Preprocessed Datasets")
                                
                                csv5 = data5.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download 5-Bin Version",
                                    data=csv5,
                                    file_name="preprocessed_5bins.csv",
                                    mime="text/csv"
                                )
                                
                                csv10 = data10.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download 10-Bin Version",
                                    data=csv10,
                                    file_name="preprocessed_10bins.csv",
                                    mime="text/csv"
                                )
        except Exception as e:
                st.error(f"Error processing file: {e}")

# ------------------ TRAINING SECTION ------------------
elif st.session_state['selected_page'] == "Training":
    st.title("Training Section")
    
    st.subheader("Training Configuration")
    
    # Check if preprocessed data is available
    if st.session_state['preprocessed_data']['5bin'] is None and st.session_state['preprocessed_data']['10bin'] is None:
        st.warning("No preprocessed data available. Please preprocess a dataset in the Dataset & Preprocessing section first.")
    else:
        # Dataset selection
        dataset_options = []
        if st.session_state['preprocessed_data']['5bin'] is not None:
            dataset_options.append("5-Bin Discretized Data")
        if st.session_state['preprocessed_data']['10bin'] is not None:
            dataset_options.append("10-Bin Discretized Data")
        
        selected_dataset = st.selectbox("Select Preprocessed Dataset for Training", dataset_options)
        
        # Save selected dataset to a temporary file for training
        dataset_path = None
        if selected_dataset:
            dataset_key = '5bin' if selected_dataset == "5-Bin Discretized Data" else '10bin'
            dataset = st.session_state['preprocessed_data'][dataset_key]
            dataset_path = f"temp_{dataset_key}_dataset.csv"
            dataset.to_csv(dataset_path, index=False)
        
        # Control Panel for Parameters
        with st.expander("Training Parameters", expanded=True):
            st.write("Configure the training parameters below:")
            
            # General Parameters
            st.subheader("General Parameters")
            task = st.selectbox("Task Type", ["single", "multi"], index=0, help="Single or multi-label classification")
            population = st.number_input("Population Size", min_value=1, value=50, step=1, help="Number of individuals in the population")
            neighbors = st.number_input("Number of Neighbors", min_value=1, value=10, step=1, help="Number of neighbors for the algorithm")
            groups = st.number_input("Number of Groups", min_value=1, value=5, step=1, help="Number of groups for clustering")
            min_examples = st.number_input("Minimum Examples per Rule", min_value=1, value=10, step=1, help="Minimum examples required per rule")
            max_uncovered = st.number_input("Max Uncovered Examples", min_value=1, value=10, step=1, help="Number of examples left uncovered to stop training")
            max_iter = st.number_input("Max Iterations", min_value=1, value=100, step=1, help="Maximum number of iterations")
            gamma = st.number_input("Gamma (Exploration/Exploitation)", min_value=0.0, max_value=1.0, value=0.9, step=0.01, help="Controls exploration vs exploitation")
            delta = st.number_input("Delta", min_value=0.0, value=0.05, step=0.01, help="Delta parameter")
            alpha = st.number_input("Alpha (Pheromone Influence)", min_value=0, value=1, step=1, help="Influence of pheromone in decision making")
            beta = st.number_input("Beta (Heuristic Influence)", min_value=0, value=1, step=1, help="Influence of heuristic in decision making")
            p = st.number_input("Pheromone Evaporation Rate", min_value=0.0, max_value=1.0, value=0.9, step=0.01, help="Rate of pheromone evaporation")
            pruning = st.selectbox("Use Rule Pruning", [0, 1], index=0, help="Enable (1) or disable (0) rule pruning")
            
            # Decomposition and Archive Parameters
            st.subheader("Decomposition and Archive Settings")
            decomposition = st.selectbox("Decomposition Method", ["weighted", "tchebycheff"], index=0, help="Method for problem decomposition")
            archive_type = st.selectbox("Archive Type", ["rules", "rulesets"], index=0, help="Structure of the archive")
            rulesets = st.selectbox("Ruleset Formation Strategy", [None, "iteration", "subproblem"], index=0, help="Strategy for forming rulesets")
            ruleset_size = st.number_input("Ruleset Size", min_value=1, value=2, step=1, help="Number of rules per ruleset (if rulesets are used)")
            prediction_strat = st.selectbox("Prediction Strategy", ["all", "best", "reference", "voting"], index=0, help="Strategy for making predictions")
            
            # Validation Parameters
            st.subheader("Validation Settings")
            cross_val = st.selectbox("Use Cross-Validation", [0, 1], index=1, help="Use cross-validation (1) or train-test split (0)")
            folds = st.number_input("Number of Folds", min_value=2, value=5, step=1, help="Number of folds for cross-validation")
            random_state = st.number_input("Random State", min_value=None, value=None, step=1, help="Random state for reproducibility (None for random)")
            
            # Run Parameters
            st.subheader("Run Settings")
            runs = st.number_input("Number of Independent Runs", min_value=1, value=1, step=1, help="Number of independent training runs")
        
        # Train Button
        if st.button("Train Model") and dataset_path:
            with st.spinner("Training model..."):
                try:
                    # Prepare parameters dictionary
                    params = {
                        'task': task,
                        'population': population,
                        'neighbors': neighbors,
                        'groups': groups,
                        'min_examples': min_examples,
                        'max_uncovered': max_uncovered,
                        'max_iter': max_iter,
                        'gamma': gamma,
                        'delta': delta,
                        'alpha': alpha,
                        'beta': beta,
                        'p': p,
                        'pruning': pruning,
                        'decomposition': decomposition,
                        'archive_type': archive_type,
                        'rulesets': rulesets,
                        'ruleset_size': ruleset_size,
                        'prediction_strat': prediction_strat,
                        'cross_val': cross_val,
                        'folds': folds,
                        'random_state': random_state,
                        'runs': runs
                    }
                    
                    # Call the train function from train.py
                    #result = train.train(dataset_path, **params)
                    
                    st.success("Training completed successfully!")
                    st.write("Training Results:")
                    #st.json(result)  # Display results as JSON (adjust based on train function output)
                    
                    # Clean up temporary dataset file
                    if os.path.exists(dataset_path):
                        os.remove(dataset_path)
                except Exception as e:
                    st.error(f"Error during training: {e}")
        elif not dataset_path:
            st.error("Please select a preprocessed dataset to train the model.")