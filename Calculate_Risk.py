import streamlit as st
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import laplace
# --- Configuration ---
st.set_page_config(
    page_title="Laplace Risk Analysis Dashboard",
    page_icon="Eq",
    layout="wide"
)
# --- Core Logic ---
def calculate_risk_metrics(alpha, scale, mean, pv):
    """
    Calculates VaR and CVaR metrics based on the provided source logic.
    """
    try:
        # Calculate VaR Rate (Log Return threshold)
        var_rate = math.log((1 - alpha) * 2) * scale + mean
        results = {
            "var_rate_raw": var_rate,
            "is_risk": False,
            "metrics": {}
        }
        # The source logic checks if the calculated rate implies a loss (negative return)
        if var_rate < 0:
            results["is_risk"] = True
            # Calculate VaR Monetary Value
            var_pt = pv * (math.exp(var_rate) - 1)
            # Calculate CVaR Rate
            cvar_rate_val = var_rate * (-1) + scale 
            # Calculate CVaR Monetary Value
            cvar_pt = pv * (math.exp(-cvar_rate_val) - 1)
            # Map to display values (reversing signs where source code did)
            results["metrics"] = {
                "VaR Rate": var_rate * -1,      # Displayed as positive magnitude
                "VaR Reserved": var_pt * -1,    # Displayed as positive magnitude
                "CVaR Rate": cvar_rate_val,     # Displayed as is
                "Mean Big Loss": cvar_pt * -1   # Displayed as positive magnitude
            }
        return results
    except ValueError:
        # Handle math domain errors (e.g. log of negative number)
        return None
# --- UI & Input Section ---
st.title("Financial Risk Analysis: Laplace Model")
st.markdown("""
This dashboard calculates **Value at Risk (VaR)** and **Conditional Value at Risk (CVaR)** assuming the data follows a **Laplace Distribution**.
""")
st.sidebar.header("Input Parameters")
# Tabbed input for Manual Entry vs CSV Upload
input_mode = st.sidebar.radio("Data Source:", ["Manual Input", "Upload Dataset (CSV)"])
if input_mode == "Upload Dataset (CSV)":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        # Select numeric column
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            target_col = st.sidebar.selectbox("Select Return/Loss Column", numeric_cols)
            data = df[target_col].dropna()
            # Auto-calculate Laplace parameters
            # MLE for Laplace: Mean = Median, Scale = Mean Absolute Deviation
            calc_mean = float(data.median())
            calc_scale = float(np.mean(np.abs(data - calc_mean)))
            st.sidebar.success(f"Est. Median: {calc_mean:.4f}")
            st.sidebar.success(f"Est. Scale: {calc_scale:.4f}")
            # Pre-fill inputs with calculated values
            mean_input = st.sidebar.number_input("Location (Median)", value=calc_mean, format="%.4f")
            scale_input = st.sidebar.number_input("Scale Parameter", value=calc_scale, format="%.4f")
        else:
            st.sidebar.error("CSV must contain numeric columns.")
            mean_input = st.sidebar.number_input("Location (Median)", value=0.0)
            scale_input = st.sidebar.number_input("Scale Parameter", value=1.0)
    else:
        st.sidebar.info("Upload a CSV to auto-calculate parameters.")
        mean_input = st.sidebar.number_input("Location (Median)", value=0.0)
        scale_input = st.sidebar.number_input("Scale Parameter", value=0.05)
else:
    # Manual Input Defaults
    scale_input = st.sidebar.number_input("Scale Parameter", value=0.02, min_value=0.0001, format="%.4f")
    mean_input = st.sidebar.number_input("Location (Median)", value=0.00, format="%.4f")
# Inputs
pv_input = st.sidebar.number_input("Present Value (PV) / Investment", value=10000.0, min_value=0.0)
alpha_input = st.sidebar.number_input("Confidence Level (Alpha)", value=0.95, min_value=0.0, format="%.2f")
# --- Calculation & Display ---
# Run calculation
result = calculate_risk_metrics(alpha_input, scale_input, mean_input, pv_input)
st.markdown("---")
if result:
    if result["is_risk"]:
        metrics = result["metrics"]
        # Display Metrics in Columns
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("VaR Rate", f":orange[{metrics['VaR Rate']:.2%}]")
        with c2:
            st.metric("VaR Reserved (Money)", f":orange[${metrics['VaR Reserved']:,.2f}]")
        with c3:
            st.metric("CVaR Rate", f":red[{metrics['CVaR Rate']:.2%}]")
        with c4:
            st.metric("Mean of Big Loss", f":red[${metrics['Mean Big Loss']:,.2f}]")
        # --- Visualization (Swapped to Plotly) ---
        st.subheader("Distribution of Returns (Laplace)")
        # Create X axis range for plotting
        x = np.linspace(mean_input - 5*scale_input, mean_input + 5*scale_input, 1000)
        y = laplace.pdf(x, loc=mean_input, scale=scale_input)
        # Create data for profit Area
        x_profit = np.linspace(0, mean_input + 5*scale_input, 1000)
        y_profit = laplace.pdf(x_profit, loc=mean_input, scale=scale_input)
        profit_values = pv_input * (np.exp(x_profit) - 1)
        # Create data for the Risk Area (Tail)
        var_cutoff = result["var_rate_raw"]
        x_tail = np.linspace(mean_input - 5*scale_input, var_cutoff, 1000)
        y_tail = laplace.pdf(x_tail, loc=mean_input, scale=scale_input)
        loss_values = pv_input * (np.exp(x_tail) - 1) * -1
        # Initialize Plotly Figure
        fig = go.Figure()
        # 1. Main Laplace Distribution Curve (Safe Area)
        fig.add_trace(go.Scatter(
            x=x, 
            y=y, 
            mode='lines',
            name='Safe Area',
            line=dict(color='#4F8BF9'),
            fill='tozeroy',  
            fillcolor='rgba(79, 139, 249, 0.1)',
            hovertemplate="Return rate: %{x:.2f}<br>PDF: %{y:.2f}<extra></extra>"
        ))
        #2. Profit area
        fig.add_trace(go.Scatter(
            x=x_profit, 
            y=y_profit, 
            mode='lines',
            name='Profit Area',
            line=dict(color="#55FF7A"),
            fill='tozeroy',  
            fillcolor='rgba(85, 255, 122, 0.1)',
            customdata = profit_values,
            hovertemplate="Profit: %{customdata:,.2f}<extra></extra>"
        ))
        # 3. Risk Area (Red Tail)
        fig.add_trace(go.Scatter(
            x=x_tail, 
            y=y_tail, 
            mode='lines',
            name='Risk Area',
            line=dict(color='#FF4B4B'),
            fill='tozeroy',
            fillcolor='rgba(255, 75, 75, 0.5)',
            customdata = loss_values,
            hovertemplate="Big Loss: %{customdata:,.2f}<extra></extra>"
        ))
        # 4. Vertical VaR Threshold Line
        fig.add_vline(
            x=var_cutoff, 
            line_width=2, 
            line_dash="dash", 
            line_color="#FF4B4B",
            annotation_text=f"VaR: {var_cutoff:.4f}", 
            annotation_position="top left"
        )
        # Layout Update
        fig.update_layout(
            title=f"Risk Visualization at {alpha_input:.1%} Confidence",
            xaxis_title="Return Rate",
            yaxis_title="Probability Density",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=80, b=40),
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
    else:
        st.success(f"At {alpha_input*100:.2f}% confidence, the calculated VaR rate is positive. No capital reservation is required according to this model.")
        st.metric("Return Rate", f"{result['var_rate_raw']:.5f} (Profit)")
else:
    st.error("Error in calculation. Please check your input parameters.")