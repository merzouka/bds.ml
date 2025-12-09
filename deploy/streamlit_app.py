import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import streamlit as st
import pandas as pd
import ipaddress
from deploy import models as models_module
from deploy.base.pipelines import *
from deploy.base.utils import *

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    """Load all models once and cache them."""
    with st.spinner("Loading models from storage..."):
        return models_module.load_models()


try:
    MODELS = load_all_models()
    if not MODELS:
        st.error("⚠️ Failed to load models. Please check your configuration.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error loading models: {str(e)}")
    st.stop()


st.title("BotNet Intrusion Detection System")
st.write(
    "This is a smart IDS for detecting the type of BotNet attack using properties of the network traffic"
)

# Display loaded models
with st.expander("ℹ️ Loaded Models", expanded=False):
    st.write(f"Successfully loaded **{len(MODELS)}** models:")
    for model_key, model_info in MODELS.items():
        st.write(f"- {model_info['name']}")

ALLOWED_PROTOCOLS = ["udp", "tcp", "icmp", "arp", "ipv6-icmp"]


# ---------- Helper validation functions ----------

def validate_ip(ip_str):
    """Check if input is valid IPv4 or IPv6."""
    try:
        ipaddress.ip_address(ip_str)
        return True
    except Exception:
        return False


def validate_port(port_str):
    """
    Validate port: allow decimal or hex.
    Returns converted integer OR None.
    """
    if port_str.strip() == "":
        return None
    try:
        if port_str.lower().startswith("0x"):
            return int(port_str, 16)
        return int(port_str)
    except Exception:
        return None


# ---------- Scrollable container ----------
scroll_container = st.container()
with scroll_container:
    st.markdown(
        """
        <style>
        .scrollable-container {
            height: 70vh;
            overflow-y: scroll;
            padding-right: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)

    with st.form("argus_form"):
        col1, col2 = st.columns(2)

        with col1:
            proto = st.text_input(
                "proto (protocol string)",
                help="Allowed values: udp, tcp, icmp, arp, ipv6-icmp"
            )
            saddr = st.text_input("saddr (source IP)")
            sport = st.text_input("sport (source port, number or hex)")
            state_number = st.number_input("state_number (numeric)", step=1)
            mean = st.number_input("mean (avg duration)", format="%.6f")
            min_v = st.number_input("min (minimum duration)", format="%.6f")
            srate = st.number_input("srate (src→dst packets/sec)", format="%.6f")
            N_IN_Conn_P_SrcIP = st.number_input(
                "N_IN_Conn_P_SrcIP (inbound connections per source IP)", step=1
            )

        with col2:
            daddr = st.text_input("daddr (destination IP)")
            dport = st.text_input("dport (destination port, number or hex)")
            seq = st.number_input("seq (sequence number)", step=1)
            stddev = st.number_input("stddev (std dev duration)", format="%.6f")
            max_v = st.number_input("max (maximum duration)", format="%.6f")
            drate = st.number_input("drate (dst→src packets/sec)", format="%.6f")
            N_IN_Conn_P_DstIP = st.number_input(
                "N_IN_Conn_P_DstIP (inbound connections per destination IP)", step=1
            )

        submitted = st.form_submit_button("Submit")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Submit handler ----------
if submitted:

    proto_clean = proto.lower().strip()
    if proto_clean not in ALLOWED_PROTOCOLS:
        st.error(f"Invalid protocol: '{proto}'. Allowed: {', '.join(ALLOWED_PROTOCOLS)}")
        st.stop()

    if not validate_ip(saddr):
        st.error(f"Invalid source IP address: {saddr}")
        st.stop()

    if not validate_ip(daddr):
        st.error(f"Invalid destination IP address: {daddr}")
        st.stop()

    sport_val = validate_port(sport)
    if sport_val is None:
        st.error(f"Invalid sport (expected decimal or hex): {sport}")
        st.stop()

    dport_val = validate_port(dport)
    if dport_val is None:
        st.error(f"Invalid dport (expected decimal or hex): {dport}")
        st.stop()

    data = {
        "proto": [proto_clean],
        "saddr": [saddr],
        "sport": [sport_val],
        "daddr": [daddr],
        "dport": [dport_val],
        "state_number": [state_number],
        "seq": [seq],
        "mean": [mean],
        "stddev": [stddev],
        "min": [min_v],
        "max": [max_v],
        "srate": [srate],
        "drate": [drate],
        "N_IN_Conn_P_SrcIP": [N_IN_Conn_P_SrcIP],
        "N_IN_Conn_P_DstIP": [N_IN_Conn_P_DstIP],
    }

    df = pd.DataFrame(data)

    st.subheader("Form Output (DataFrame)")
    st.dataframe(df, use_container_width=True)

    # ---------- Model Predictions ----------
    st.subheader("Model Predictions")
    
    # Progress bar for predictions
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    predictions_data = []
    total_models = len(MODELS)
    
    for idx, (model_key, model_info) in enumerate(MODELS.items()):
        try:
            status_text.text(f"Running {model_info['name']}...")
            progress_bar.progress((idx + 1) / total_models)
            
            # Get numeric prediction
            prediction = model_info['model'].predict(df)
            prediction_value = prediction[0] if len(prediction) > 0 else None
            
            # Try to get category label
            category = None
            try:
                if hasattr(model_info['model'], 'category'):
                    categories = model_info['model'].category(prediction)
                    category = categories[0] if len(categories) > 0 else "Unknown"
            except Exception:
                category = "N/A"
            
            predictions_data.append({
                "Model": model_info['name'],
                "Numeric Prediction": prediction_value,
                "Category": category if category else "N/A"
            })
            
        except Exception as e:
            predictions_data.append({
                "Model": model_info['name'],
                "Numeric Prediction": "Error",
                "Category": f"Error: {str(e)}"
            })
    
    # Clear progress indicators
    status_text.empty()
    progress_bar.empty()
    
    # Display predictions as a table
    predictions_df = pd.DataFrame(predictions_data)
    st.dataframe(predictions_df, use_container_width=True)
    
    # Display individual predictions with color coding
    st.markdown("---")
    st.subheader("Detailed Predictions")
    
    cols = st.columns(2)
    for idx, pred in enumerate(predictions_data):
        col = cols[idx % 2]
        with col:
            with st.container():
                st.markdown(f"**{pred['Model']}**")
                
                # Color code based on category
                category = pred['Category']
                if category == "Normal":
                    color = "green"
                elif "Error" in str(category):
                    color = "orange"
                elif category == "N/A":
                    color = "gray"
                else:
                    color = "red"
                
                st.markdown(
                    f"<span style='color:{color}; font-size:18px;'>"
                    f"**{category}**</span> (Code: {pred['Numeric Prediction']})",
                    unsafe_allow_html=True
                )
                st.markdown("---")
