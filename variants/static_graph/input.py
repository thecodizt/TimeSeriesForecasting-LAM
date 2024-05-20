import streamlit as st
import pandas as pd

def input():
    node_data = None
    edge_data = None
    
    with st.sidebar:
        node_data_file = st.file_uploader(
            label="Node Data",
            type=["csv"]
        )
        
        edge_data_file = st.file_uploader(
            label="Edge Data",
            type=["csv"]
        )
        
        if node_data_file:
            node_data = pd.read_csv(node_data_file)
        
        if edge_data_file:
            edge_data = pd.read_csv(edge_data_file)
        
    return node_data, edge_data