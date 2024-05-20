import random
import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
from sklearn.metrics import mean_squared_error

def generate_adjancency_matrix(num_nodes, density = 0.5, allow_self_edge=False):
    adj_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]
    
    current_density = 0
    
    while current_density < density:
        while True:
            i = random.randint(0,num_nodes-1)
            j = random.randint(0,num_nodes-1)
            
            if not adj_matrix[i][j]:
                adj_matrix[i][j] = 1
                current_density += 1/(num_nodes*num_nodes)
                break
    
    if not allow_self_edge:
        for i in range(num_nodes):
            adj_matrix[i][i] = 0
        
    return adj_matrix

def generate_control_points(num_points = 10):
    control_points = []
    
    while len(control_points) < num_points:
        control_points.append(random.random())
        
    return control_points

def generate_spline(control_points, n_points=100, noise=0):
    # Create x values for control points
    x = np.linspace(0, 1, len(control_points))

    # Create cubic spline
    cs = CubicSpline(x, control_points)

    # Generate N evenly spaced x values between 0 and 1
    x_new = np.linspace(0, 1, n_points)

    # Compute y values for these x values
    y_new = cs(x_new)
    
    # Introduce noise
    if noise != 0:
        noise_amount = np.random.normal(0, noise, size=y_new.shape)
        y_new = y_new + noise_amount

    return y_new

def generate_node_data(num_properties=1, num_records=100, num_control_points=10, noise=0):
    node_data = []
    
    while len(node_data) < num_properties:
        node_property = generate_spline(generate_control_points(num_control_points), num_records, noise)
        node_data.append(node_property)
        
    tranposed = [list(i) for i in zip(*node_data)]
        
    return tranposed
        
def flatten_dataframe(df):
    df_flat = df.reset_index().melt(id_vars='timestamp', var_name='feature', value_name='value')
    return df_flat

def unflatten_dataframe(df_flat):
    df = df_flat.pivot(index='timestamp', columns='feature', values='value')
    df.reset_index(drop=True, inplace=True)
    df.columns.name = None
    return df

def merge_melted_dfs(dfs):
    # Add 'entity' column to each DataFrame and concatenate them
    for i, df in enumerate(dfs):
        df['node'] = i
    
    df_concat = pd.concat(dfs, ignore_index=True)
    return df_concat

def array_to_dataframe(array_2d):
    df = pd.DataFrame(array_2d)
    return df

def generate_n_node_flat_data(num_nodes, num_records, num_properties, num_control_points, noise):
    flat_dfs = []
    
    while (len(flat_dfs)) < num_nodes:
        generated_node_data = generate_node_data(num_properties, num_records, num_control_points, noise)
        generated_node_data = array_to_dataframe(generated_node_data)
        flat_df = flatten_dataframe(generated_node_data)
        flat_dfs.append(flat_df)
        
    return merge_melted_dfs(flat_dfs)

def visualize_adjacency_matrix(adj_matrix):
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(np.array(adj_matrix))

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color='black')

    # Use streamlit to display the graph
    st.pyplot(plt)
    
def get_node_data_from_merged(merged_data, node_index):
    filtered = merged_data[merged_data["node"] == node_index]
    filtered.drop(["node"], axis=1, inplace=True)
    
    unflattened = unflatten_dataframe(filtered)
    
    return unflattened

def visualize_forecast(df_train, df_actual, df_predicted, columns):
    # Define a list of colors for the plots
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors if needed

    # Plot the actual vs predicted values
    plt.figure(figsize=(10, 6))
    for i, column in enumerate(columns):
        plt.plot(range(len(df_train)), df_train[column], color=colors[i % len(colors)], alpha=0.3, label=f'Training {column}')  # Darker shade for training data
        plt.plot(range(len(df_train), len(df_train) + len(df_actual)), df_actual[column], color=colors[i % len(colors)], alpha=0.6, label=f'Actual {column}')  # Lighter shade for actual values
        plt.plot(range(len(df_train), len(df_train) + len(df_predicted)), df_predicted[column], color=colors[i % len(colors)], alpha=1, label=f'Predicted {column}')  # Lighter shade for predicted values

    plt.title('Training, Actual vs Predicted')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    st.pyplot(plt)  # Display the plot in Streamlit
    
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))