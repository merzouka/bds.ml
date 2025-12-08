import sys
from pathlib import Path

# Add the parent directory (botnetds/) to Python path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import streamlit as st
import pandas as pd
import ipaddress
import .models


st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("BotNet Intrusion Detection System")
st.write(
    "This is a smart IDS for detecting the type of BotNet attack using properties of the network traffic"
)

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
    st.dataframe(df)

    # model.predict(df)  <-- add later
