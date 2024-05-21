import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

from variants.static_graph.input import input
from variants.static_graph.models import prophet_forecast, arima_forecast, ets_forecast, multivariate_prophet_forecast, multivariate_var_forecast, multivariate_varmax_forecast
from utils import get_node_data_from_merged

def static_graph():
    st.title("Static Graph Forecasting")
    
    forecast_type = None
    forecast_type_options = ["Univariate", "Multivariate - Node Level", "Multivariate K-hop"]
    
    with st.sidebar:
        forecast_type = st.selectbox("Forecast Type", options=forecast_type_options)
    
    node_data, edge_data = input()
    
    merged_data = node_data
    
    st.header(forecast_type + " Forecasting")
    
    if node_data is not None and edge_data is not None:
    
        if forecast_type == forecast_type_options[0]:
            
            num_nodes = len(node_data["node"].unique())
            
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

                plt.title('Actual vs Predicted')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot(plt)  # Display the plot in Streamlit
                
                # Plot the actual vs predicted values
                plt.figure(figsize=(10, 6))
                for i, column in enumerate(test.columns):
                    if column != 'ds':
                        plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                        plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

                plt.title('Actual vs Predicted')
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
                
                # Plot the actual vs predicted values
                plt.figure(figsize=(10, 6))
                for i, column in enumerate(test.columns):
                    if column != 'ds':
                        plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                        plt.plot(range(len(train), len(train) + len(test)), forecasts[column][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

                plt.title('Actual vs Predicted')
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
                
                # Plot the actual vs predicted values
                plt.figure(figsize=(10, 6))
                for i, column in enumerate(test.columns):
                    if column != 'ds':
                        plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                        plt.plot(range(len(train), len(train) + len(test)), forecasts[column][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

                plt.title('Actual vs Predicted')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot(plt)  # Display the plot in Streamlit

        if forecast_type == forecast_type_options[1]:
            
            num_nodes = len(node_data["node"].unique())
            
            with st.expander(label="Multivariate Prophet", expanded=True):
                results = {}
                rmse_values = []
                
                for node_index_forecast in range(num_nodes):
                    node_data_forecast = get_node_data_from_merged(merged_data=merged_data, node_index=node_index_forecast)
                    rmse, train, test, forecasts = multivariate_prophet_forecast(df=node_data_forecast)
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
                
                # Plot the actual vs predicted values
                plt.figure(figsize=(10, 6))
                for i, column in enumerate(test.columns):
                    if column != 'ds':
                        plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                        plt.plot(range(len(train), len(train) + len(test)), forecasts[column]['yhat'][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

                plt.title('Actual vs Predicted')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot(plt)  # Display the plot in Streamlit
                
            with st.expander(label="Multivariate VAR", expanded=True):
                results = {}
                rmse_values = []
                
                for node_index_forecast in range(num_nodes):
                    node_data_forecast = get_node_data_from_merged(merged_data=merged_data, node_index=node_index_forecast)
                    rmse, train, test, forecasts = multivariate_var_forecast(df=node_data_forecast)
                    results[node_index_forecast] = {'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                    rmse_values.append(rmse)
                
                avg_rmse = np.mean(rmse_values)
                st.metric(f'Average RMSE', avg_rmse)
                
                node_index = st.slider(label="Node ID", min_value=0, max_value=max(num_nodes-1, 1), key='mul_var')
                node_data = get_node_data_from_merged(merged_data=merged_data, node_index=node_index)
                
                st.subheader("Node Data")
                st.dataframe(node_data)

                result = results[node_index]
                train = result['train']
                test = result['test']
                forecasts = result['forecasts']
                
                # Define a list of colors for the plots
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed

                # Convert the numpy array to a DataFrame
                forecasts_df = pd.DataFrame(forecasts, columns=train.columns)

                # Plot the actual vs predicted values
                plt.figure(figsize=(10, 6))
                for i, column in enumerate(test.columns):
                    if column != 'ds':
                        plt.plot(range(len(train)), train[column], color=colors[i % len(colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                        plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                        plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

                plt.title('Training, Actual vs Predicted')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot(plt)  # Display the plot in Streamlit
                
                # Plot the actual vs predicted values
                plt.figure(figsize=(10, 6))
                for i, column in enumerate(test.columns):
                    if column != 'ds':
                        plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                        plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

                plt.title('Actual vs Predicted')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot(plt)  # Display the plot in Streamlit
                
            with st.expander(label="Multivariate VARMAX", expanded=True):
                results = {}
                rmse_values = []
                
                for node_index_forecast in range(num_nodes):
                    node_data_forecast = get_node_data_from_merged(merged_data=merged_data, node_index=node_index_forecast)
                    rmse, train, test, forecasts = multivariate_varmax_forecast(df=node_data_forecast)
                    results[node_index_forecast] = {'rmse': rmse, 'train': train, 'test': test, 'forecasts': forecasts}
                    rmse_values.append(rmse)
                
                avg_rmse = np.mean(rmse_values)
                st.metric(f'Average RMSE', avg_rmse)
                
                node_index = st.slider(label="Node ID", min_value=0, max_value=max(num_nodes-1, 1), key='mul_varmax')
                node_data = get_node_data_from_merged(merged_data=merged_data, node_index=node_index)
                
                st.subheader("Node Data")
                st.dataframe(node_data)

                result = results[node_index]
                train = result['train']
                test = result['test']
                forecasts = result['forecasts']
                
                # Define a list of colors for the plots
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed

                # Convert the numpy array to a DataFrame
                forecasts_df = pd.DataFrame(forecasts, columns=train.columns)

                # Plot the actual vs predicted values
                plt.figure(figsize=(10, 6))
                for i, column in enumerate(test.columns):
                    if column != 'ds':
                        plt.plot(range(len(train)), train[column], color=colors[i % len(colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
                        plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
                        plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

                plt.title('Training, Actual vs Predicted')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot(plt)  # Display the plot in Streamlit
                
                # Plot the actual vs predicted values
                plt.figure(figsize=(10, 6))
                for i, column in enumerate(test.columns):
                    if column != 'ds':
                        plt.plot(range(len(train), len(train) + len(test)), test[column], color=colors[i % len(colors)], alpha=0.3, label=f'Actual {column}')  # Lighter shade for actual values
                        plt.plot(range(len(train), len(train) + len(test)), forecasts_df[column][-len(test):], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

                plt.title('Actual vs Predicted')
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.legend()
                st.pyplot(plt)  # Display the plot in Streamlit
              