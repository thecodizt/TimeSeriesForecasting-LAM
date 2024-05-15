import streamlit as st

from variants import static_graph, dynamic_graph

def main():
    graph_type = None
    
    with st.sidebar:
        st.header("Time Series Analysis and Forecasting")
        
        graph_type_labels = [
            "Static Graph",
            "Dynamic Graph"
        ]
        
        graph_type = st.selectbox("Graph Type", options=graph_type_labels)
        
    if graph_type == graph_type_labels[0]:
        static_graph()
        
    if graph_type == graph_type_labels[1]:
        dynamic_graph()
    
if __name__=="__main__":
    main()