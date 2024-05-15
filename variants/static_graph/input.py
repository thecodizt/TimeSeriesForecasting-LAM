import streamlit as st

def input():
    with st.sidebar:
        num_nodes = st.number_input(label="Number of Nodes in Graph", min_value=1, step=1)
        num_records = st.number_input(label="Number of records for each node", min_value=1, step=10)
        num_prop = st.number_input(label="Number of properties for each node", min_value=1, step=1)
        edge_density = st.number_input(label="Edge Density in Adjacency Matrix", min_value=0.0, max_value=1.0, step=0.05)
        noise = st.number_input(label="Maximum Noise in Values", min_value=0.0, max_value=1.0, step=0.05)
        num_control_points = st.number_input(label="Number of Control Points in Generation", min_value=2, step=1)
        
    return num_nodes, num_records, num_prop, edge_density, noise, num_control_points