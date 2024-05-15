import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

from variants.static_graph.input import input
from variants.static_graph.models import prophet_forecast, arima_forecast, ets_forecast
from utils import generate_adjancency_matrix, generate_n_node_flat_data, visualize_adjacency_matrix, get_node_data_from_merged


def static_graph():
    st.title("Static Graph Forecasting")
    
    forecast_type = None
    forecast_type_options = ["Univariate", "Multivariate - Node Level", "Multivariate K-hop"]
    
    with st.sidebar:
        forecast_type = st.selectbox("Forecast Type", options=forecast_type_options)
    
    num_nodes, num_records, num_properties, edge_density, noise, num_control_points = input()
    
    # Generate the data
    adj_matrix = generate_adjancency_matrix(num_nodes=num_nodes, density=edge_density)
    merged_data = generate_n_node_flat_data(num_nodes=num_nodes, num_records=num_records, num_control_points=num_control_points, num_properties=num_properties, noise=noise)
    
    with st.expander(label="Adjacency Matrix of Graph Edges", expanded=True):
        mat_str = ""
        
        for l in adj_matrix:
            for i in l:
                mat_str += str(i) + " "
            mat_str += "\n"
        
        st.code(mat_str)
        
    with st.expander(label="Graph Structure from Adjacency Matrix", expanded=True):
        visualize_adjacency_matrix(adj_matrix=adj_matrix)
        
    with st.expander(label="Generated Input Data", expanded=True):
        st.dataframe(merged_data)
        
    with st.expander(label="Visualize Generated Data", expanded=True):
        node_index_vis = st.slider(label="Node ID", min_value=0, max_value=max(num_nodes-1, 1), key="vis_node")
        node_data_vis = get_node_data_from_merged(merged_data=merged_data, node_index=node_index_vis)
        
        # Draw the graph
        plt.figure(figsize=(8, 6))
        plt.plot(node_data_vis)

        st.subheader("Line Plot")
        # Use streamlit to display the graph
        st.pyplot(plt)
        
    st.header(forecast_type + " Forecasting")
    
    with st.expander(label="Prophet", expanded=True):
        results = {}
        rmse_values = []
        
        for node_index_forecast in range(num_nodes):
            node_data_forecast = get_node_data_from_merged(merged_data=merged_data, node_index=node_index_forecast)
            rmse, train, test, forecasts = prophet_forecast(df=node_data_forecast)
            results[node_index_forecast] = {'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
            rmse_values.append(rmse)
        
        avg_rmse = np.mean(rmse_values)
        st.metric(f'Average RMSE', avg_rmse)
        
        node_index = st.slider(label="Node ID", min_value=0, max_value=max(num_nodes-1, 1))
        node_data = get_node_data_from_merged(merged_data=merged_data, node_index=node_index)
        
        st.subheader("Node Data")
        st.dataframe(node_data)

        result = results[node_index]
        train = result['train']
        test = result['test']
        forecasts = result['forecasts']
        
        # Define a list of colors for the plots
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed

        # Plot the actual vs predicted values
        plt.figure(figsize=(10, 6))
        for i, column in enumerate(test.columns):
            if column != 'ds':
                plt.plot(range(len(train)), train[column], color=colors[i % len(colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

        plt.title('Training, Actual vs Predicted')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        st.pyplot(plt)  # Display the plot in Streamlit

    with st.expander(label="ARIMA", expanded=True):
        results = {}
        rmse_values = []
        
        for node_index_forecast in range(num_nodes):
            node_data_forecast = get_node_data_from_merged(merged_data=merged_data, node_index=node_index_forecast)
            rmse, train, test, forecasts = arima_forecast(df=node_data_forecast)
            results[node_index_forecast] = {'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
            rmse_values.append(rmse)
        
        avg_rmse = np.mean(rmse_values)
        st.metric(f'Average RMSE', avg_rmse)
        
        node_index = st.slider(label="Node ID", min_value=0, max_value=max(num_nodes-1, 1), key="arima")
        node_data = get_node_data_from_merged(merged_data=merged_data, node_index=node_index)
        
        st.subheader("Node Data")
        st.dataframe(node_data)

        result = results[node_index]
        train = result['train']
        test = result['test']
        forecasts = result['forecasts']
        
        # Define a list of colors for the plots
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed

        # Plot the actual vs predicted values
        plt.figure(figsize=(10, 6))
        for i, column in enumerate(test.columns):
            if column != 'ds':
                plt.plot(range(len(train)), train[column], color=colors[i % len(colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                plt.plot(range(len(train), len(train) + len(test)), forecasts[column][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

        plt.title('Training, Actual vs Predicted')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        st.pyplot(plt)  # Display the plot in Streamlit

    with st.expander(label="Exponential Smoothing", expanded=True):
        results = {}
        rmse_values = []
        
        for node_index_forecast in range(num_nodes):
            node_data_forecast = get_node_data_from_merged(merged_data=merged_data, node_index=node_index_forecast)
            rmse, train, test, forecasts = ets_forecast(df=node_data_forecast)
            results[node_index_forecast] = {'rmse ': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
            rmse_values.append(rmse)
        
        rmse_mape = np.mean(rmse_values)
        st.metric(f'Average RMSE', rmse_mape)
        
        node_index = st.slider(label="Node ID", min_value=0, max_value=max(num_nodes-1, 1), key="ets")
        node_data = get_node_data_from_merged(merged_data=merged_data, node_index=node_index)
        
        st.subheader("Node Data")
        st.dataframe(node_data)

        result = results[node_index]
        train = result['train']
        test = result['test']
        forecasts = result['forecasts']
        
        # Define a list of colors for the plots
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed

        # Plot the actual vs predicted values
        plt.figure(figsize=(10, 6))
        for i, column in enumerate(test.columns):
            if column != 'ds':
                plt.plot(range(len(train)), train[column], color=colors[i % len(colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                plt.plot(range(len(train), len(train) + len(test)), forecasts[column][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

        plt.title('Training, Actual vs Predicted')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        st.pyplot(plt)  # Display the plot in Streamlit
