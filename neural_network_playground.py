import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QPushButton, 
                             QSlider, QSpinBox, QComboBox, QTextEdit, 
                             QGroupBox, QTabWidget, QSplitter, QFrame, QCheckBox, QSizePolicy, QFileDialog, QMessageBox, QDoubleSpinBox, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QPalette, QColor
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from matplotlib.colors import LinearSegmentedColormap
from typing import List
import json

# Quantization functions
def quantize_tensor(tensor, bits):
    """Quantize a tensor to specified number of bits"""
    if bits == 32:  # No quantization
        return tensor
    
    # Calculate quantization parameters
    min_val = tensor.min()
    max_val = tensor.max()
    
    if bits == 1:
        # Binary quantization: -1 or 1
        return torch.sign(tensor)
    elif bits == 1.58:
        # Ternary quantization: -1, 0, 1
        scale = max(abs(min_val), abs(max_val))
        if scale == 0:
            return tensor
        normalized = tensor / scale
        # Use threshold-based quantization for ternary
        threshold = 0.5
        quantized = torch.where(normalized > threshold, 1.0, 
                               torch.where(normalized < -threshold, -1.0, 0.0))
        return quantized * scale
    elif bits == 2:
        # 2-bit quantization: -1, -0.33, 0.33, 1
        scale = max(abs(min_val), abs(max_val))
        if scale == 0:
            return tensor
        normalized = tensor / scale
        quantized = torch.round(normalized * 1.5) / 1.5
        return quantized * scale
    else:
        # n-bit quantization
        num_levels = 2 ** bits
        scale = max(abs(min_val), abs(max_val))
        if scale == 0:
            return tensor
        normalized = tensor / scale
        quantized = torch.round(normalized * (num_levels - 1)) / (num_levels - 1)
        return quantized * scale

def apply_quantization(model, bits):
    """Apply quantization to all parameters in the model"""
    if bits == 32:  # No quantization
        return model
    
    # Use no_grad to prevent this from interfering with autograd
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                # Use copy_() for a safer in-place operation that is compatible with the optimizer state
                param.data.copy_(quantize_tensor(param.data, bits))
    return model

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', dropout_rate=0.0):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout_layers = []  # Use regular list to allow None values
        
        if len(hidden_sizes) == 0:
            # No hidden layers - direct connection from input to output
            self.layers.append(nn.Linear(input_size, output_size))
        else:
            # Input layer to first hidden layer
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            if dropout_rate > 0:
                self.dropout_layers.append(nn.Dropout(dropout_rate))
            else:
                self.dropout_layers.append(None)
            
            # Hidden layers
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                if dropout_rate > 0:
                    self.dropout_layers.append(nn.Dropout(dropout_rate))
                else:
                    self.dropout_layers.append(None)
            
            # Output layer
            self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
            self.dropout_layers.append(None)  # No dropout on output layer
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        if len(self.layers) == 1:
            # No hidden layers - no activation on output
            return self.layers[0](x)
        else:
            # Apply activation and dropout to all layers except the last one
            for i, layer in enumerate(self.layers[:-1]):
                x = layer(x)
                x = self.activation(x)
                if self.dropout_layers[i] is not None:
                    x = self.dropout_layers[i](x)
            x = self.layers[-1](x)
            return x

class NetworkVisualizer(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f0f0f0')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
    def visualize_network(self, layer_sizes, weights=None):
        self.axes.clear()
        if not layer_sizes:
            return

        G = nx.DiGraph()
        pos = {}
        node_colors = []

        # Create nodes
        for layer_idx, size in enumerate(layer_sizes):
            x_pos = layer_idx * 2
            for node_idx in range(size):
                y_pos = (node_idx - (size-1)/2.0)
                node_id = f"L{layer_idx}_N{node_idx}"
                G.add_node(node_id)
                pos[node_id] = (x_pos, y_pos)
                if layer_idx == 0:
                    node_colors.append('#aabbcc') # Input color
                elif layer_idx == len(layer_sizes) - 1:
                    node_colors.append('#ccbbaa') # Output color
                else:
                    node_colors.append('lightblue')

        # Create edges
        edge_colors = []
        edge_widths = []
        for layer_idx in range(len(layer_sizes) - 1):
            weight_tensor = None
            if weights:
                weight_tensor = weights.get(f'layers.{layer_idx}.weight')

            for from_node in range(layer_sizes[layer_idx]):
                for to_node in range(layer_sizes[layer_idx + 1]):
                    from_id = f"L{layer_idx}_N{from_node}"
                    to_id = f"L{layer_idx + 1}_N{to_node}"
                    G.add_edge(from_id, to_id)
                    
                    if weight_tensor is not None:
                        w = weight_tensor[to_node, from_node].item()
                        edge_colors.append('#1f77b4' if w > 0 else '#ff7f0e')
                        edge_widths.append(min(abs(w) * 2, 4) + 0.5)
                    else:
                        edge_colors.append('gray')
                        edge_widths.append(1)
        
        max_nodes = max(layer_sizes) if layer_sizes else 1
        self.axes.set_ylim(-max_nodes/2.0 -1, max_nodes/2.0 + 1)
        self.axes.set_xlim(-1, (len(layer_sizes) - 1) * 2 + 1)

        nx.draw_networkx_nodes(G, pos, ax=self.axes, node_shape='s', node_color=node_colors, node_size=200) # type: ignore
        nx.draw_networkx_edges(G, pos, ax=self.axes, edge_color=edge_colors, width=edge_widths, alpha=0.6, arrows=False) # type: ignore

        self.axes.set_title('Network Architecture')
        self.axes.axis('off')
        self.draw()

class DataVisualizer(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f0f0f0')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Custom colormap from orange to blue
        self.custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#ff7f0e", "#f0f0f0", "#1f77b4"])
        
        # Color palette for multiple classes
        self.class_colors = ['#1f77b4', '#ff7f0e', '#4CAF50', '#ff0000', '#9400D3', '#8B4513', '#FFC0CB', '#808080', '#00FFFF', '#FFFF00']

    def plot_data(self, X_train, y_train, X_test=None, y_test=None, show_test_data=False, predictions=None):
        self.axes.clear()
        
        if y_train is None:
            self.draw()
            return

        # Get unique classes and create color mapping
        unique_classes = np.unique(y_train)
        num_classes = len(unique_classes)
        
        # Plot training data with proper colors
        for i, class_label in enumerate(unique_classes):
            mask = y_train == class_label
            color = self.class_colors[i % len(self.class_colors)]
            self.axes.scatter(X_train[mask, 0], X_train[mask, 1], c=color, alpha=0.7, s=20, label=f'Class {class_label}')
        
        # Plot test data if requested
        if show_test_data and X_test is not None and y_test is not None:
            for i, class_label in enumerate(unique_classes):
                mask = y_test == class_label
                if np.any(mask):  # Only plot if there are test points for this class
                    color = self.class_colors[i % len(self.class_colors)]
                    self.axes.scatter(X_test[mask, 0], X_test[mask, 1], c=color, alpha=0.7, s=20, marker='x')

        # Plot decision boundaries if predictions are provided
        if predictions is not None and num_classes > 1:
            x_min, x_max = -6, 6
            y_min, y_max = -6, 6
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                               np.linspace(y_min, y_max, 50))
            
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            grid_predictions = predictions(grid_points)
            
            if grid_predictions.size > 0:
                grid_predictions = grid_predictions.reshape(xx.shape)
                
                if num_classes == 2:
                    # Binary classification - show probability of class 1
                    self.axes.contourf(xx, yy, grid_predictions, alpha=0.5, cmap=self.custom_cmap, levels=50)
                else:
                    # Multi-class classification - show decision boundaries
                    # Create a discrete colormap for class regions
                    from matplotlib.colors import ListedColormap
                    discrete_cmap = ListedColormap(self.class_colors[:num_classes])
                    self.axes.contourf(xx, yy, grid_predictions, alpha=0.3, cmap=discrete_cmap, levels=num_classes)
        
        # Set plot limits based on the data
        if X_train is not None and len(X_train) > 0:
            x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
            y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
            self.axes.set_xlim(x_min, x_max)
            self.axes.set_ylim(y_min, y_max)
        else:
            # Default limits if no data
            self.axes.set_xlim(-2.5, 2.5)
            self.axes.set_ylim(-2.5, 2.5)

        # Plot decision boundary if predictions are available
        if predictions and X_train is not None:
            x_min, x_max = self.axes.get_xlim()
        
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_aspect('equal', adjustable='box')
        
        # Add legend for multiple classes - position it outside the plot
        if num_classes > 1:
            self.axes.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        
        self.fig.tight_layout()
        self.draw()

class LossGraphVisualizer(FigureCanvas):
    def __init__(self, parent=None, width=6, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f0f0f0')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_losses(self, train_losses, test_losses):
        self.axes.clear()
        self.axes.plot(train_losses, label='Training Loss', color='#1f77b4')
        self.axes.plot(test_losses, label='Test Loss', color='#ff7f0e')
        self.axes.set_title('Loss Over Epochs')
        self.axes.set_xlabel('Epoch')
        self.axes.set_ylabel('Loss')
        self.axes.legend()
        self.axes.grid(True, linestyle='--', alpha=0.6)
        self.fig.tight_layout()
        self.draw()

class PaintCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f0f0f0')
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Data storage
        self.points = {}
        self.current_class = 0
        self.drawing = False
        self.current_button = None  # Track current mouse button
        self.current_mouse_pos = None  # Track current mouse position
        self.last_mouse_pos = None  # Track last mouse position for movement-based plotting
        
        # Timer for point placement
        self.drawing_timer = QTimer()
        self.drawing_timer.timeout.connect(self.place_point_timer)
        
        # Movement threshold for movement-based plotting
        self.movement_threshold = 0.1  # Minimum distance to trigger point placement
        
        # Setup the canvas
        self.setup_canvas()
        self.connect_events()
        
    def setup_canvas(self):
        # Get limits dynamically from DataVisualizer
        x_min, x_max = -6, 6  # Default, will be updated dynamically
        y_min, y_max = -6, 6  # Default, will be updated dynamically
        
        self.axes.set_xlim(x_min, x_max)
        self.axes.set_ylim(y_min, y_max)
        self.axes.set_aspect('equal')
        self.axes.grid(True, alpha=0.3)
        
        # Initialize with default classes if not already done
        if not self.points:
            self.points = {'class_0': [], 'class_1': [], 'class_2': [], 'class_3': [], 'class_4': []}
        
        self.update_title()
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.draw()
        
    def sync_limits_with_data_visualizer(self, data_visualizer):
        """Sync the canvas limits with the data visualizer"""
        if data_visualizer and hasattr(data_visualizer, 'axes'):
            x_min, x_max = data_visualizer.axes.get_xlim()
            y_min, y_max = data_visualizer.axes.get_ylim()
            self.axes.set_xlim(x_min, x_max)
            self.axes.set_ylim(y_min, y_max)
            self.draw()
        
    def update_title(self):
        """Update the title with current point counts"""
        total_count = sum(len(points) for points in self.points.values())
        
        # Get the selected class
        selected_class = 0
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'class_selector'):
            selected_class = self.main_window.class_selector.currentIndex()
        
        class_colors = ['Blue', 'Orange', 'Green', 'Red', 'Purple', 'Brown', 'Pink', 'Gray', 'Cyan', 'Yellow']
        selected_color = class_colors[selected_class] if selected_class < len(class_colors) else 'Unknown'
        
        title = f'Draw Data Points (Total: {total_count})\n'
        title += f'Active Class: {selected_class} ({selected_color}) - Click to draw\n'
        
        # Show class counts dynamically
        class_info = []
        for class_key in sorted(self.points.keys()):
            class_index = int(class_key.split('_')[1])
            color_name = class_colors[class_index] if class_index < len(class_colors) else 'Unknown'
            count = len(self.points[class_key])
            class_info.append(f'Class {class_index} ({color_name}): {count}')
        
        title += ' | '.join(class_info)
        
        if total_count < 10:
            title += f'\n⚠️ Need at least 10 points (currently {total_count})'
        else:
            title += f'\n✅ Ready to generate data!'
            
        self.axes.set_title(title, fontsize=10)
        
    def connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
    def on_click(self, event):
        if event.inaxes != self.axes:
            return
            
        self.drawing = True
        self.current_button = event.button
        self.current_mouse_pos = (event.xdata, event.ydata)
        self.last_mouse_pos = (event.xdata, event.ydata)
        
        # Get the selected class from the main window
        selected_class = 0  # Default
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'class_selector'):
            selected_class = self.main_window.class_selector.currentIndex()
        
        # Check if movement-based plotting is enabled
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'movement_based_checkbox'):
            if self.main_window.movement_based_checkbox.isChecked():
                # Movement-based: place initial point and don't start timer
                self.add_point(event.xdata, event.ydata, selected_class)
                return
        
        # Timer-based: start the timer
        self.start_drawing_timer()
        
        # Place initial point
        self.add_point(event.xdata, event.ydata, selected_class)
        
    def on_motion(self, event):
        if not self.drawing or event.inaxes != self.axes:
            return
            
        # Update current mouse position
        self.current_mouse_pos = (event.xdata, event.ydata)
        
        # Get the selected class from the main window
        selected_class = 0  # Default
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'class_selector'):
            selected_class = self.main_window.class_selector.currentIndex()
        
        # Check if movement-based plotting is enabled
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'movement_based_checkbox'):
            if self.main_window.movement_based_checkbox.isChecked():
                # Movement-based: check if mouse moved enough to place a point
                if self.last_mouse_pos and self.current_mouse_pos:
                    dx = self.current_mouse_pos[0] - self.last_mouse_pos[0]
                    dy = self.current_mouse_pos[1] - self.last_mouse_pos[1]
                    distance = (dx**2 + dy**2)**0.5
                    
                    if distance >= self.movement_threshold:
                        self.add_point(event.xdata, event.ydata, selected_class)
                        self.last_mouse_pos = (event.xdata, event.ydata)
        
    def on_release(self, event):
        self.drawing = False
        self.current_button = None
        self.current_mouse_pos = None
        self.last_mouse_pos = None
        
        # Stop the timer
        self.drawing_timer.stop()
        
    def start_drawing_timer(self):
        """Start the timer for point placement"""
        # Get speed setting from main window
        speed = 10  # Default speed
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'drawing_speed_spinbox'):
            speed = self.main_window.drawing_speed_spinbox.value()
        
        # Limit maximum speed to prevent lag
        speed = min(speed, 50)  # Cap at 50 points per second for smooth performance
        
        # Convert speed to milliseconds (higher speed = shorter interval)
        interval_ms = max(20, int(1000 / speed))  # Minimum 20ms interval
        self.drawing_timer.start(interval_ms)
        
    def place_point_timer(self):
        """Timer callback to place points"""
        if self.drawing and self.current_mouse_pos:
            x, y = self.current_mouse_pos
            
            # Get the selected class from the main window
            selected_class = 0  # Default
            if hasattr(self, 'main_window') and hasattr(self.main_window, 'class_selector'):
                selected_class = self.main_window.class_selector.currentIndex()
            
            self.add_point(x, y, selected_class)
            
    def add_point(self, x, y, class_index):
        if x is None or y is None:
            return
            
        # Determine class key based on class index
        class_key = f'class_{class_index}'
        
        # Add point to storage
        self.points[class_key].append([x, y])
        
        # Redraw the entire canvas efficiently
        self.redraw_canvas()
        
    def redraw_canvas(self):
        """Efficiently redraw the entire canvas"""
        # Store current limits before clearing
        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        
        self.axes.clear()
        
        # Restore the limits
        self.axes.set_xlim(x_min, x_max)
        self.axes.set_ylim(y_min, y_max)
        self.axes.set_aspect('equal')
        self.axes.grid(True, alpha=0.3)
        
        # Define colors for classes - same as DataVisualizer
        colors = ['#1f77b4', '#ff7f0e', '#4CAF50', '#ff0000', '#9400D3', '#8B4513', '#FFC0CB', '#808080', '#00FFFF', '#FFFF00']
        
        # Draw all points efficiently
        for class_key in sorted(self.points.keys()):
            if self.points[class_key]:  # Only draw if there are points
                class_index = int(class_key.split('_')[1])
                color = colors[class_index] if class_index < len(colors) else '#000000'
                class_points = np.array(self.points[class_key])
                self.axes.scatter(class_points[:, 0], class_points[:, 1], 
                                 c=color, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        self.update_title()
        self.axes.set_xlabel('X')
        self.axes.set_ylabel('Y')
        self.draw()
        
    def clear_canvas(self):
        # Clear all classes dynamically
        self.points = {}
        if hasattr(self, 'main_window') and hasattr(self.main_window, 'class_selector'):
            for i in range(self.main_window.class_selector.count()):
                self.points[f'class_{i}'] = []
        else:
            # Fallback to default classes
            self.points = {'class_0': [], 'class_1': [], 'class_2': [], 'class_3': [], 'class_4': []}
        
        self.axes.clear()
        self.setup_canvas()
        
    def get_data(self):
        """Return the drawn data as numpy arrays"""
        # Check if any class has points
        has_points = any(len(points) > 0 for points in self.points.values())
        if not has_points:
            return None, None
            
        X = []
        y = []
        
        # Add points from all classes
        for class_key in sorted(self.points.keys()):
            class_index = int(class_key.split('_')[1])
            for point in self.points[class_key]:
                X.append(point)
                y.append(class_index)
            
        return np.array(X), np.array(y)
        
    def load_data(self, X, y):
        """Load existing data into the canvas"""
        self.clear_canvas()
        
        for i, (x, y_val) in enumerate(zip(X, y)):
            class_key = f'class_{int(y_val)}'
            self.points[class_key].append([x[0], x[1]])
            
        self.redraw_canvas()

class TrainingThread(QThread):
    progress_updated = pyqtSignal(int, float, float, float)
    training_finished = pyqtSignal()
    
    def __init__(self, model, X_train, y_train, X_test, y_test, learning_rate, quantization_bits=32.0,
                 optimizer_name='Adam', batch_size=32, weight_decay=0.0, momentum=0.9, 
                 beta1=0.9, beta2=0.999, scheduler_name='None', scheduler_param=0.1,
                 early_stopping=False, early_stop_patience=50, dropout_rate=0.0):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.learning_rate = learning_rate
        self.quantization_bits = quantization_bits
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.scheduler_name = scheduler_name
        self.scheduler_param = scheduler_param
        self.early_stopping = early_stopping
        self.early_stop_patience = early_stop_patience
        self.dropout_rate = dropout_rate
        self.running = True
        self.train_losses = []
        self.test_losses = []
        
    def run(self):
        # Move model and data to GPU if available
        self.model = self.model.to(device)
        X_train = self.X_train.to(device)
        y_train = self.y_train.to(device)
        X_test = self.X_test.to(device)
        y_test = self.y_test.to(device)

        criterion = nn.CrossEntropyLoss()
        
        # Create optimizer based on selection
        if self.optimizer_name == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, 
                                 weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))
        elif self.optimizer_name == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, 
                                weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        elif self.optimizer_name == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, 
                                  weight_decay=self.weight_decay, betas=(self.beta1, self.beta2))
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create learning rate scheduler
        scheduler = None
        if self.scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(self.scheduler_param))
        elif self.scheduler_name == 'ExponentialLR':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_param)
        elif self.scheduler_name == 'CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.scheduler_param))
        
        # Early stopping variables
        best_test_loss = float('inf')
        patience_counter = 0
        
        epoch = 0
        while self.running:
            epoch += 1
            
            # Mini-batch training
            self.model.train()
            total_train_loss = 0.0
            num_batches = 0
            
            # Create mini-batches
            for i in range(0, len(X_train), self.batch_size):
                batch_end = min(i + self.batch_size, len(X_train))
                X_batch = X_train[i:batch_end]
                y_batch = y_train[i:batch_end]
                
                optimizer.zero_grad()
                train_outputs = self.model(X_batch)
                train_loss = criterion(train_outputs, y_batch)
                train_loss.backward()
                optimizer.step()
                
                total_train_loss += train_loss.item()
                num_batches += 1
            
            avg_train_loss = total_train_loss / num_batches if num_batches > 0 else 0.0
            
            # Apply quantization after optimizer step
            if self.quantization_bits != 32:
                apply_quantization(self.model, self.quantization_bits)
            
            # Calculate test loss
            self.model.eval()
            with torch.no_grad():
                test_outputs = self.model(X_test)
                test_loss = criterion(test_outputs, y_test)

            # Calculate accuracy on training data
            self.model.train()
            with torch.no_grad():
                train_outputs = self.model(X_train)
                _, predicted = torch.max(train_outputs.data, 1)
                accuracy = (predicted == y_train).sum().item() / y_train.size(0)
            
            self.progress_updated.emit(epoch, avg_train_loss, test_loss.item(), accuracy)
            
            self.train_losses.append(avg_train_loss)
            self.test_losses.append(test_loss.item())
            
            # Update learning rate scheduler
            if scheduler:
                scheduler.step()
            
            # Early stopping check
            if self.early_stopping:
                if test_loss.item() < best_test_loss:
                    best_test_loss = test_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Move model back to CPU after training
        self.model = self.model.to('cpu')
        self.training_finished.emit()
    
    def stop(self):
        self.running = False

class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title, parent=None):
        # Pass an empty string to QGroupBox to avoid drawing the default title
        super().__init__("", parent)
        self.is_collapsed = False
        self.content_widget = None
        self.toggle_button = None
        self._title = title
        self.setup_ui()

    def setup_ui(self):
        # Main layout for this group box
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        # Header widget to contain button and title
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)

        # Create toggle button
        self.toggle_button = QPushButton("▼")
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                font-size: 14px;
                font-weight: bold;
                color: #0078d7;
            }
            QPushButton:hover {
                color: #005a9e;
            }
        """)
        self.toggle_button.clicked.connect(self.toggle_collapse)

        # Create title label
        title_label = QLabel(self._title)
        title_label.setStyleSheet("font-weight: bold; font-size: 11pt; color: #333;")

        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        self.main_layout.addWidget(header_widget)

        # Create content widget
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(10, 5, 10, 5) # Add padding to content
        self.content_layout.setSpacing(8) # Increase spacing between content items

        self.main_layout.addWidget(self.content_widget)

        # Apply a cleaner border style
        self.setStyleSheet("""
            CollapsibleGroupBox {
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #ffffff;
                margin-top: 0px; /* Remove extra margin */
            }
        """)
        
        # Set initial state
        self.set_collapsed(False)

    def toggle_collapse(self):
        self.set_collapsed(not self.is_collapsed)

    def set_collapsed(self, collapsed):
        self.is_collapsed = collapsed
        if self.content_widget:
            self.content_widget.setVisible(not collapsed)

        if collapsed:
            if self.toggle_button:
                self.toggle_button.setText("▶")
            self.setMaximumHeight(40) # Set a fixed collapsed height
        else:
            if self.toggle_button:
                self.toggle_button.setText("▼")
            self.setMaximumHeight(16777215) # Unset max height
    
    def addWidget(self, widget):
        """Add a widget to the content area"""
        self.content_layout.addWidget(widget)
    
    def addLayout(self, layout):
        """Add a layout to the content area"""
        self.content_layout.addLayout(layout)
    
    def addSpacing(self, spacing):
        """Add spacing to the content area"""
        self.content_layout.addSpacing(spacing)

class NeuralNetworkPlayground(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.training_thread = None
        self.train_losses = []
        self.test_losses = []
        
        self.network_visualizer = NetworkVisualizer()
        self.network_visualizer.setMinimumHeight(200) # Ensure it has enough space
        self.network_visualizer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.data_visualizer = DataVisualizer()
        self.loss_visualizer = LossGraphVisualizer()
        self.paint_canvas = PaintCanvas()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('Neural Network Playground')
        self.setGeometry(100, 100, 1600, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QWidget {
                background-color: #f0f0f0;
                color: #333;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                /* This style is now handled by CollapsibleGroupBox */
            }
            QGroupBox::title {
                /* This is now handled by a QLabel inside CollapsibleGroupBox */
            }
            QPushButton {
                background-color: #e7e7e7;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #d7d7d7;
            }
            QPushButton:pressed {
                background-color: #c7c7c7;
            }
            QSlider::groove:horizontal {
                border: 1px solid #ccc;
                height: 8px;
                background: #e7e7e7;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #005a9e;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSpinBox, QComboBox {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QLabel {
                font-size: 10pt;
            }
            #EpochLabel, #LossLabel, #AccuracyLabel {
                font-size: 12pt;
                font-weight: bold;
            }
            #TitleLabel {
                font-size: 16pt;
                font-weight: bold;
                color: #0078d7;
            }
        """)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # --- Playground Tab ---
        playground_widget = QWidget()
        playground_layout = QVBoxLayout(playground_widget)
        self.tabs.addTab(playground_widget, "Playground")

        # Top bar for controls
        top_bar_widget = self.create_top_bar()
        playground_layout.addWidget(top_bar_widget)
        
        # Main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        playground_layout.addWidget(content_splitter)

        # Left panel for data, features, and layers
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        left_panel_layout.setContentsMargins(10, 10, 10, 10)  # Add padding around content
        left_panel_layout.setSpacing(5)  # Reduce spacing between items within groups
        
        # Create scroll area for left panel
        left_scroll_area = QScrollArea()
        left_scroll_area.setWidget(left_panel)
        left_scroll_area.setWidgetResizable(True)
        left_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll_area.setMinimumWidth(400)
        left_scroll_area.setMaximumWidth(500)
        left_scroll_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f8f8f8;
            }
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        content_splitter.addWidget(left_scroll_area)
        
        data_group = self.create_data_group()
        features_group = self.create_features_group()
        learning_params_group = self.create_learning_params_group()
        layers_group = self.create_layers_group()

        left_panel_layout.addWidget(data_group)
        left_panel_layout.addSpacing(10)  # Add spacing between groups
        left_panel_layout.addWidget(features_group)
        left_panel_layout.addSpacing(10)  # Add spacing between groups
        left_panel_layout.addWidget(learning_params_group)
        left_panel_layout.addSpacing(10)  # Add spacing between groups
        left_panel_layout.addWidget(layers_group)
        left_panel_layout.addStretch()

        # Right panel for output visualization
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        content_splitter.addWidget(right_panel)

        output_group = CollapsibleGroupBox("OUTPUT")
        output_group.set_collapsed(False) # Make sure it's expanded by default
        
        # Add summary stats (moved from top bar)
        stats_layout = QHBoxLayout()
        self.loss_label = QLabel("Test loss: 0.000")
        self.accuracy_label = QLabel("Training loss: 0.000")
        stats_layout.addWidget(self.loss_label)
        stats_layout.addSpacing(20)
        stats_layout.addWidget(self.accuracy_label)
        stats_layout.addStretch()
        output_group.addLayout(stats_layout)
        output_group.addSpacing(10)

        output_group.addWidget(self.data_visualizer)
        
        # Add checkboxes
        checkbox_layout = QHBoxLayout()
        self.show_test_data_check = QCheckBox("Show test data")
        self.show_test_data_check.stateChanged.connect(self.update_data_visualization)
        checkbox_layout.addWidget(self.show_test_data_check)
        output_group.addLayout(checkbox_layout)
        
        output_group.addWidget(self.loss_visualizer)

        right_panel_layout.addWidget(output_group)
        content_splitter.setSizes([500, 700])  # Give more space to left panel for scrollable content

        # --- Custom Data Tab ---
        custom_widget = QWidget()
        custom_layout = QVBoxLayout(custom_widget)
        self.tabs.addTab(custom_widget, "Custom Data")

        # Paint canvas (make it expand)
        self.paint_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.paint_canvas.main_window = self  # Give access to main window
        custom_layout.addWidget(self.paint_canvas, stretch=1)
        
        # Control buttons and settings
        controls_layout = QVBoxLayout()
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Drawing Speed:"))
        self.drawing_speed_spinbox = QSpinBox()
        self.drawing_speed_spinbox.setRange(1, 999999)  # Very high limit, practically unlimited
        self.drawing_speed_spinbox.setValue(10)
        self.drawing_speed_spinbox.setToolTip("Higher values = more points when dragging")
        speed_layout.addWidget(self.drawing_speed_spinbox)
        speed_layout.addWidget(QLabel("(1+)"))
        controls_layout.addLayout(speed_layout)
        
        # Movement-based plotting checkbox
        movement_layout = QHBoxLayout()
        self.movement_based_checkbox = QCheckBox("Movement-based plotting")
        self.movement_based_checkbox.setToolTip("Place points based on mouse movement instead of timer")
        self.movement_based_checkbox.setChecked(False)
        self.movement_based_checkbox.stateChanged.connect(self.on_movement_mode_changed)
        movement_layout.addWidget(self.movement_based_checkbox)
        movement_layout.addStretch()
        controls_layout.addLayout(movement_layout)
        
        # Class selector
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Active Class:"))
        self.class_selector = QComboBox()
        self.class_selector.addItems(['Class 0 (Blue)', 'Class 1 (Orange)', 'Class 2 (Green)', 'Class 3 (Red)', 'Class 4 (Purple)'])
        self.class_selector.setCurrentIndex(0)
        self.class_selector.currentIndexChanged.connect(self.on_class_changed)
        class_layout.addWidget(self.class_selector)
        
        # Add/Remove class buttons
        add_class_button = QPushButton("+")
        add_class_button.setFixedSize(30, 30)
        add_class_button.setToolTip("Add a new class")
        add_class_button.clicked.connect(self.add_class)
        class_layout.addWidget(add_class_button)
        
        remove_class_button = QPushButton("-")
        remove_class_button.setFixedSize(30, 30)
        remove_class_button.setToolTip("Remove last class")
        remove_class_button.clicked.connect(self.remove_class)
        class_layout.addWidget(remove_class_button)
        
        class_layout.addStretch()
        controls_layout.addLayout(class_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        clear_button = QPushButton("Clear Canvas")
        clear_button.clicked.connect(self.paint_canvas.clear_canvas)
        button_layout.addWidget(clear_button)

        save_button = QPushButton("Save Data")
        save_button.clicked.connect(self.save_custom_data)
        button_layout.addWidget(save_button)
        
        load_button = QPushButton("Load Data")
        load_button.clicked.connect(self.load_custom_data)
        button_layout.addWidget(load_button)
        
        button_layout.addStretch()
        controls_layout.addLayout(button_layout)
        
        custom_layout.addLayout(controls_layout)

        # Initial data
        self.generate_data()
        self.create_model()

    def create_top_bar(self):
        top_bar_widget = QWidget()
        layout = QHBoxLayout(top_bar_widget)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(15)

        # === Left side: Run and Reset Buttons ===
        self.run_button = QPushButton("▶")
        self.run_button.setCheckable(True)
        self.run_button.toggled.connect(self.toggle_training)
        self.run_button.setFixedSize(40, 40)
        self.run_button.setStyleSheet("font-size: 20px;")
        
        self.reset_button = QPushButton("↺")
        self.reset_button.clicked.connect(self.generate_data)
        self.reset_button.setFixedSize(40, 40)
        self.reset_button.setStyleSheet("font-size: 20px;")
        
        layout.addWidget(self.run_button)
        layout.addWidget(self.reset_button)
        
        layout.addSpacing(10)

        # === Save/Load Buttons ===
        save_button = QPushButton("Save Config")
        save_button.clicked.connect(self.save_configuration)
        layout.addWidget(save_button)

        load_button = QPushButton("Load Config")
        load_button.clicked.connect(self.load_configuration)
        layout.addWidget(load_button)

        layout.addSpacing(20)

        # === Middle: Main Hyperparameters ===
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        # Row 0: Labels
        grid_layout.addWidget(QLabel("Learning rate"), 0, 0)
        grid_layout.addWidget(QLabel("Activation"), 0, 1)
        grid_layout.addWidget(QLabel("Quantization"), 0, 2)
        
        # Row 1: Controls
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(['0.0001', '0.0003', '0.001', '0.003', '0.01', '0.03', '0.1', '0.3', '1'])
        self.lr_combo.setCurrentText('0.03')
        grid_layout.addWidget(self.lr_combo, 1, 0)

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(['ReLU', 'Tanh', 'Sigmoid', 'Softmax'])
        grid_layout.addWidget(self.activation_combo, 1, 1)

        self.quantization_combo = QComboBox()
        self.quantization_combo.addItems(['32-bit (None)', '1-bit', '1.58-bit', '2-bit', '3-bit', '4-bit', '8-bit', '16-bit'])
        self.quantization_combo.setCurrentText('32-bit (None)')
        self.quantization_combo.currentTextChanged.connect(self.create_model)
        grid_layout.addWidget(self.quantization_combo, 1, 2)

        layout.addLayout(grid_layout)
        layout.addStretch()

        # === Right side: Status Info ===
        status_layout = QHBoxLayout()
        status_layout.setSpacing(20)

        # Epoch Counter
        epoch_layout = QVBoxLayout()
        epoch_layout.addWidget(QLabel("Epoch"))
        self.epoch_label = QLabel("000,000")
        self.epoch_label.setObjectName("EpochLabel")
        epoch_layout.addWidget(self.epoch_label)
        status_layout.addLayout(epoch_layout)

        # GPU Status
        gpu_layout = QVBoxLayout()
        gpu_layout.addWidget(QLabel("Device"))
        gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
        gpu_color = "#4CAF50" if torch.cuda.is_available() else "#FF9800"
        self.gpu_label = QLabel(gpu_status)
        self.gpu_label.setStyleSheet(f"color: {gpu_color}; font-weight: bold; font-size: 12pt;")
        gpu_layout.addWidget(self.gpu_label)
        status_layout.addLayout(gpu_layout)

        layout.addLayout(status_layout)

        return top_bar_widget

    def create_data_group(self):
        group = CollapsibleGroupBox("DATA")
        
        # Dataset selection
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(QLabel("Dataset:"))
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['Moons', 'Circles', 'Linear', 'Custom'])
        self.dataset_combo.currentTextChanged.connect(self.generate_data)
        dataset_layout.addWidget(self.dataset_combo)
        group.addLayout(dataset_layout)

        # Standard dataset controls
        standard_layout = QGridLayout()
        standard_layout.addWidget(QLabel("Noise:"), 0, 0)
        self.noise_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_slider.setRange(0, 50)
        self.noise_slider.setValue(20)
        self.noise_slider.valueChanged.connect(self.generate_data)
        standard_layout.addWidget(self.noise_slider, 0, 1)
        standard_layout.addWidget(QLabel("Train/Test Ratio:"), 1, 0)
        self.ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.ratio_slider.setRange(1, 9)
        self.ratio_slider.setValue(5)
        self.ratio_slider.valueChanged.connect(self.generate_data)
        standard_layout.addWidget(self.ratio_slider, 1, 1)
        group.addLayout(standard_layout)
        return group

    def create_features_group(self):
        group = CollapsibleGroupBox("FEATURES")
        
        # Add visualization update frequency controls
        viz_label = QLabel("Visualization Update Frequency (epochs):")
        viz_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        group.addWidget(viz_label)
        
        # Loss plot update frequency
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Loss Plot:"))
        self.loss_update_spinbox = QSpinBox()
        self.loss_update_spinbox.setRange(1, 1000)
        self.loss_update_spinbox.setValue(10)
        self.loss_update_spinbox.setToolTip("Update loss plot every N epochs")
        self.loss_update_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 3px;
                min-width: 60px;
            }
        """)
        loss_layout.addWidget(self.loss_update_spinbox)
        group.addLayout(loss_layout)
        
        # Data visualization update frequency
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Data Plot:"))
        self.data_update_spinbox = QSpinBox()
        self.data_update_spinbox.setRange(1, 1000)
        self.data_update_spinbox.setValue(50)
        self.data_update_spinbox.setToolTip("Update data visualization with predictions every N epochs")
        self.data_update_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 3px;
                min-width: 60px;
            }
        """)
        data_layout.addWidget(self.data_update_spinbox)
        group.addLayout(data_layout)
        
        # Network visualization update frequency
        network_layout = QHBoxLayout()
        network_layout.addWidget(QLabel("Network:"))
        self.network_update_spinbox = QSpinBox()
        self.network_update_spinbox.setRange(1, 1000)
        self.network_update_spinbox.setValue(100)
        self.network_update_spinbox.setToolTip("Update network visualization every N epochs")
        self.network_update_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 3px;
                padding: 3px;
                min-width: 60px;
            }
        """)
        network_layout.addWidget(self.network_update_spinbox)
        group.addLayout(network_layout)
        
        return group

    def create_learning_params_group(self):
        group = CollapsibleGroupBox("LEARNING PARAMETERS")
        
        # Optimizer selection
        optimizer_layout = QHBoxLayout()
        optimizer_layout.addWidget(QLabel("Optimizer:"))
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['Adam', 'SGD', 'RMSprop', 'Adagrad', 'AdamW'])
        self.optimizer_combo.setCurrentText('Adam')
        self.optimizer_combo.currentTextChanged.connect(self.on_optimizer_changed)
        optimizer_layout.addWidget(self.optimizer_combo)
        group.addLayout(optimizer_layout)
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spinbox = QSpinBox()
        self.batch_size_spinbox.setRange(1, 1000)
        self.batch_size_spinbox.setValue(32)
        self.batch_size_spinbox.setToolTip("Mini-batch size for training")
        batch_layout.addWidget(self.batch_size_spinbox)
        group.addLayout(batch_layout)
        
        # Weight decay (L2 regularization)
        weight_decay_layout = QHBoxLayout()
        weight_decay_layout.addWidget(QLabel("Weight Decay:"))
        self.weight_decay_spinbox = QDoubleSpinBox()
        self.weight_decay_spinbox.setRange(0.0, 1.0)
        self.weight_decay_spinbox.setValue(0.0)
        self.weight_decay_spinbox.setSingleStep(0.001)
        self.weight_decay_spinbox.setDecimals(4)
        self.weight_decay_spinbox.setToolTip("L2 regularization strength")
        weight_decay_layout.addWidget(self.weight_decay_spinbox)
        group.addLayout(weight_decay_layout)
        
        # Momentum (for SGD)
        self.momentum_layout = QHBoxLayout()
        self.momentum_layout.addWidget(QLabel("Momentum:"))
        self.momentum_spinbox = QDoubleSpinBox()
        self.momentum_spinbox.setRange(0.0, 1.0)
        self.momentum_spinbox.setValue(0.9)
        self.momentum_spinbox.setSingleStep(0.1)
        self.momentum_spinbox.setDecimals(2)
        self.momentum_spinbox.setToolTip("Momentum for SGD optimizer")
        self.momentum_layout.addWidget(self.momentum_spinbox)
        group.addLayout(self.momentum_layout)
        
        # Beta1 (for Adam)
        self.beta1_layout = QHBoxLayout()
        self.beta1_layout.addWidget(QLabel("Beta1:"))
        self.beta1_spinbox = QDoubleSpinBox()
        self.beta1_spinbox.setRange(0.0, 0.9999) # Must be < 1.0
        self.beta1_spinbox.setValue(0.9)
        self.beta1_spinbox.setSingleStep(0.01)
        self.beta1_spinbox.setDecimals(4)
        self.beta1_spinbox.setToolTip("Beta1 parameter for Adam optimizer (must be < 1.0)")
        self.beta1_layout.addWidget(self.beta1_spinbox)
        group.addLayout(self.beta1_layout)
        
        # Beta2 (for Adam)
        self.beta2_layout = QHBoxLayout()
        self.beta2_layout.addWidget(QLabel("Beta2:"))
        self.beta2_spinbox = QDoubleSpinBox()
        self.beta2_spinbox.setRange(0.0, 0.9999) # Must be < 1.0
        self.beta2_spinbox.setValue(0.999)
        self.beta2_spinbox.setSingleStep(0.001)
        self.beta2_spinbox.setDecimals(4)
        self.beta2_spinbox.setToolTip("Beta2 parameter for Adam optimizer (must be < 1.0)")
        self.beta2_layout.addWidget(self.beta2_spinbox)
        group.addLayout(self.beta2_layout)
        
        # Learning rate scheduler
        scheduler_layout = QHBoxLayout()
        scheduler_layout.addWidget(QLabel("LR Scheduler:"))
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(['None', 'StepLR', 'ExponentialLR', 'CosineAnnealingLR'])
        self.scheduler_combo.setCurrentText('None')
        self.scheduler_combo.currentTextChanged.connect(self.on_scheduler_changed)
        scheduler_layout.addWidget(self.scheduler_combo)
        group.addLayout(scheduler_layout)
        
        # Scheduler parameters
        self.scheduler_params_layout = QHBoxLayout()
        self.scheduler_params_layout.addWidget(QLabel("Scheduler Param:"))
        self.scheduler_param_spinbox = QDoubleSpinBox()
        self.scheduler_param_spinbox.setRange(0.1, 10.0)
        self.scheduler_param_spinbox.setValue(0.1)
        self.scheduler_param_spinbox.setSingleStep(0.1)
        self.scheduler_param_spinbox.setDecimals(2)
        self.scheduler_param_spinbox.setToolTip("Parameter for learning rate scheduler")
        self.scheduler_params_layout.addWidget(self.scheduler_param_spinbox)
        group.addLayout(self.scheduler_params_layout)
        
        # Early stopping
        early_stop_layout = QHBoxLayout()
        early_stop_layout.addWidget(QLabel("Early Stopping:"))
        self.early_stop_checkbox = QCheckBox("Enable")
        self.early_stop_checkbox.setChecked(False)
        early_stop_layout.addWidget(self.early_stop_checkbox)
        
        early_stop_layout.addWidget(QLabel("Patience:"))
        self.early_stop_patience_spinbox = QSpinBox()
        self.early_stop_patience_spinbox.setRange(1, 1000)
        self.early_stop_patience_spinbox.setValue(50)
        self.early_stop_patience_spinbox.setToolTip("Number of epochs to wait before early stopping")
        early_stop_layout.addWidget(self.early_stop_patience_spinbox)
        group.addLayout(early_stop_layout)
        
        # Dropout
        dropout_layout = QHBoxLayout()
        dropout_layout.addWidget(QLabel("Dropout:"))
        self.dropout_spinbox = QDoubleSpinBox()
        self.dropout_spinbox.setRange(0.0, 0.9)
        self.dropout_spinbox.setValue(0.0)
        self.dropout_spinbox.setSingleStep(0.1)
        self.dropout_spinbox.setDecimals(2)
        self.dropout_spinbox.setToolTip("Dropout probability for regularization")
        dropout_layout.addWidget(self.dropout_spinbox)
        group.addLayout(dropout_layout)
        
        # Set initial visibility based on optimizer
        self.on_optimizer_changed()
        self.on_scheduler_changed()
        
        return group

    def create_layers_group(self):
        group = CollapsibleGroupBox("HIDDEN LAYERS")

        group.addWidget(self.network_visualizer)

        self.hidden_layers_layout = QVBoxLayout()
        self.hidden_layers_layout.setSpacing(8) # Add spacing between layer rows
        group.addLayout(self.hidden_layers_layout)
        
        group.addSpacing(10)

        # Add Layer button centered
        add_layer_button_layout = QHBoxLayout()
        add_layer_button_layout.addStretch()
        add_layer_button = QPushButton("Add Layer")
        add_layer_button.setFixedWidth(150) # Give it a reasonable size
        add_layer_button.clicked.connect(self.add_hidden_layer)
        add_layer_button_layout.addWidget(add_layer_button)
        add_layer_button_layout.addStretch()
        group.addLayout(add_layer_button_layout)

        self.hidden_layer_widgets = []
        self.add_hidden_layer(neurons=4)
        self.add_hidden_layer(neurons=2)

        return group

    def add_hidden_layer(self, neurons=3):
        layer_layout = QHBoxLayout()
        layer_layout.setSpacing(10)
        
        label = QLabel(f"Layer {len(self.hidden_layer_widgets) + 1}")
        label.setFixedWidth(60) # Give label a fixed width for alignment
        
        neuron_spinbox = QSpinBox()
        neuron_spinbox.setRange(1, 9999)
        neuron_spinbox.setValue(neurons)
        neuron_spinbox.setStyleSheet("""
            QSpinBox {
                background-color: #ffffff;
                border: 1px solid #ccc;
                border-radius: 4px; /* Less rounded for a cleaner look */
                padding: 5px;
                font-size: 11pt;
            }
        """)
        # Connect the valueChanged signal to recreate the model
        neuron_spinbox.valueChanged.connect(self.create_model)
        
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: self.remove_hidden_layer(layer_layout))

        layer_layout.addWidget(label)
        layer_layout.addWidget(neuron_spinbox)
        layer_layout.addStretch() # Pushes the remove button to the right
        layer_layout.addWidget(remove_button)

        self.hidden_layers_layout.addLayout(layer_layout)
        self.hidden_layer_widgets.append(layer_layout)
        # create_model() will handle the visualization update
        self.create_model()

    def remove_hidden_layer(self, layer_layout):
        if layer_layout in self.hidden_layer_widgets:
            # Block signals from spinboxes to prevent updates while deleting
            spinbox = layer_layout.itemAt(1).widget()
            if spinbox:
                spinbox.blockSignals(True)

            self.hidden_layer_widgets.remove(layer_layout)
            
            # Remove all widgets from the layout
            while layer_layout.count():
                item = layer_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            
            # Remove the layout itself
            self.hidden_layers_layout.removeItem(layer_layout)
            layer_layout.deleteLater()
            
            # Update layer labels
            for i, layout in enumerate(self.hidden_layer_widgets):
                label = layout.itemAt(0).widget()
                if label:
                    label.setText(f"Layer {i + 1}")

            # create_model() will handle the visualization update
            self.create_model()

    def toggle_training(self, checked):
        if checked:
            self.start_training()
            self.run_button.setText("⏹")
        else:
            self.stop_training()
            self.run_button.setText("▶")

    def generate_data(self):
        self.stop_training() # Stop any ongoing training
        
        dataset_type = self.dataset_combo.currentText()
        noise = self.noise_slider.value() / 100.0
        ratio = self.ratio_slider.value() / 10.0
        n_samples = 1000
        
        X, y = None, None

        if dataset_type == 'Moons':
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
        elif dataset_type == 'Circles':
            X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=42)
        elif dataset_type == 'Linear':
            X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, 
                                       n_informative=2, random_state=42, n_clusters_per_class=1, class_sep=2)
        elif dataset_type == 'Custom':
            X_custom, y_custom = self.paint_canvas.get_data()
            if X_custom is None or len(X_custom) < 10:
                QMessageBox.warning(self, "Not Enough Data", "Please draw at least 10 points for the custom dataset.")
                # Fallback to a default dataset
                X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
                self.dataset_combo.setCurrentText("Moons")
            else:
                X, y = X_custom, y_custom
                print(f"✅ Using custom dataset with {len(X)} points")
                if y is not None:
                    unique_classes, counts = np.unique(y, return_counts=True)
                    n_train = int(len(X) * ratio)
                    n_test = len(X) - n_train
                    print(f"   Training: {n_train} points, Test: {n_test} points")
                    class_counts_str = ", ".join([f"Class {c}: {count}" for c, count in zip(unique_classes, counts)])
                    print(f"   ({class_counts_str})")
        
        if X is None or y is None:
             X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

        # Data preprocessing
        X_scaled = StandardScaler().fit_transform(X)
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_scaled, y, test_size=(1-ratio), random_state=42)
        
        # Convert to tensors immediately to ensure correct types throughout the class
        self.X_train = torch.tensor(X_train_np, dtype=torch.float32)
        self.y_train = torch.tensor(y_train_np, dtype=torch.long)
        self.X_test = torch.tensor(X_test_np, dtype=torch.float32)
        self.y_test = torch.tensor(y_test_np, dtype=torch.long)

        # Clear old losses and update visualizations
        self.train_losses.clear()
        self.test_losses.clear()
        self.loss_visualizer.plot_losses([], [])
        self.update_data_visualization()
        self.create_model()

    def create_model(self):
        if self.X_train is None or self.y_train is None:
            return
            
        hidden_sizes = []
        for layout in self.hidden_layer_widgets:
            spinbox = layout.itemAt(1).widget()
            if spinbox:
                hidden_sizes.append(spinbox.value())
        
        input_size = self.X_train.shape[1]
        output_size = len(torch.unique(self.y_train))
        
        # Get activation function
        activation_map = {'ReLU': 'relu', 'Tanh': 'tanh', 'Sigmoid': 'sigmoid', 'Softmax': 'softmax'}
        activation = activation_map.get(self.activation_combo.currentText(), 'relu')
        
        # Get dropout rate
        dropout_rate = self.dropout_spinbox.value()
        
        self.model = NeuralNetwork(input_size, hidden_sizes, output_size, activation, dropout_rate)
        self.update_network_visualization()

    def start_training(self):
        if self.model and self.X_train is not None and self.X_test is not None:
            learning_rate = float(self.lr_combo.currentText())
            
            # Get quantization bits from UI
            quantization_text = self.quantization_combo.currentText()
            if quantization_text == '32-bit (None)':
                quantization_bits = 32
            elif quantization_text == '1.58-bit':
                quantization_bits = 1.58
            else:
                quantization_bits = int(quantization_text.split('-')[0])
            
            # Get all learning parameters
            optimizer_name = self.optimizer_combo.currentText()
            batch_size = self.batch_size_spinbox.value()
            weight_decay = self.weight_decay_spinbox.value()
            momentum = self.momentum_spinbox.value()
            beta1 = self.beta1_spinbox.value()
            beta2 = self.beta2_spinbox.value()
            scheduler_name = self.scheduler_combo.currentText()
            scheduler_param = self.scheduler_param_spinbox.value()
            early_stopping = self.early_stop_checkbox.isChecked()
            early_stop_patience = self.early_stop_patience_spinbox.value()
            dropout_rate = self.dropout_spinbox.value()
            
            self.training_thread = TrainingThread(
                self.model, self.X_train, self.y_train, 
                self.X_test, self.y_test, learning_rate, quantization_bits,
                optimizer_name, batch_size, weight_decay, momentum,
                beta1, beta2, scheduler_name, scheduler_param,
                early_stopping, early_stop_patience, dropout_rate
            )
            self.training_thread.progress_updated.connect(self.update_training_progress)
            self.training_thread.training_finished.connect(self.training_finished)
            self.training_thread.start()

    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.run_button.setChecked(False)

    def update_data_visualization(self):
        if self.X_train is not None and self.y_train is not None and self.X_test is not None and self.y_test is not None:
            show_test = self.show_test_data_check.isChecked()
            self.data_visualizer.plot_data(self.X_train.numpy(), self.y_train.numpy(), 
                                           self.X_test.numpy(), self.y_test.numpy(), 
                                           show_test_data=show_test)
            # Sync paint canvas limits with data visualizer
            self.paint_canvas.sync_limits_with_data_visualizer(self.data_visualizer)

    def update_training_progress(self, epoch, train_loss, test_loss, accuracy):
        # Update labels every epoch (lightweight)
        self.epoch_label.setText(f"{epoch:,}")
        self.loss_label.setText(f"Test loss: {test_loss:.4f}")
        self.accuracy_label.setText(f"Training loss: {train_loss:.4f}")

        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)

        # Update visualizations based on user-defined frequencies
        # Update loss plot
        if epoch % self.loss_update_spinbox.value() == 0:
            self.loss_visualizer.plot_losses(self.train_losses, self.test_losses)
        
        # Update data visualization with predictions (expensive operation)
        if epoch % self.data_update_spinbox.value() == 0:
            if self.X_train is None or self.y_train is None:
                return
            
            def get_predictions(x_grid):
                if self.model is None:
                    return np.array([])
                
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(x_grid, dtype=torch.float32)
                    # Move input tensor to the same device as the model
                    x_tensor = x_tensor.to(next(self.model.parameters()).device)
                    logits = self.model(x_tensor)
                    
                    # For visualization, we want predicted class labels
                    if logits.shape[1] > 1:  # Multi-class
                        predicted_classes = torch.argmax(logits, dim=1)
                        predictions_np = predicted_classes.cpu().numpy()
                    else:  # Binary classification
                        predicted_probs = nn.Sigmoid()(logits)
                        predictions_np = predicted_probs.cpu().numpy().flatten()
                
                self.model.train()
                return predictions_np

            show_test = self.show_test_data_check.isChecked()
            if self.X_train is not None and self.y_train is not None and self.X_test is not None and self.y_test is not None:
                self.data_visualizer.plot_data(self.X_train.numpy(), self.y_train.numpy(), 
                                               self.X_test.numpy(), self.y_test.numpy(), 
                                               show_test_data=show_test, 
                                               predictions=get_predictions)
        
        # Update network visualization (most expensive)
        if epoch % self.network_update_spinbox.value() == 0:
            if self.model:
                self.update_network_visualization(self.model.state_dict())

    def training_finished(self):
        print("Training finished.")
        self.run_button.setChecked(False)
        self.run_button.setText("▶")
        
        # Final update to show the trained model state
        if self.model and self.X_train is not None and self.y_train is not None:
            self.update_network_visualization(self.model.state_dict())
            
            # Final prediction visualization
            def get_predictions(x_grid):
                if self.model is None:
                    return np.array([])
                
                self.model.eval()
                with torch.no_grad():
                    x_tensor = torch.tensor(x_grid, dtype=torch.float32)
                    # Move input tensor to the same device as the model
                    x_tensor = x_tensor.to(next(self.model.parameters()).device)
                    logits = self.model(x_tensor)
                    
                    # For visualization, we want predicted class labels
                    if logits.shape[1] > 1:  # Multi-class
                        predicted_classes = torch.argmax(logits, dim=1)
                        predictions_np = predicted_classes.cpu().numpy()
                    else:  # Binary classification
                        predicted_probs = nn.Sigmoid()(logits)
                        predictions_np = predicted_probs.cpu().numpy().flatten()
                
                self.model.train()
                return predictions_np

            show_test = self.show_test_data_check.isChecked()
            if self.X_train is not None and self.y_train is not None and self.X_test is not None and self.y_test is not None:
                self.data_visualizer.plot_data(self.X_train.numpy(), self.y_train.numpy(), 
                                               self.X_test.numpy(), self.y_test.numpy(), 
                                               show_test_data=show_test, 
                                               predictions=get_predictions)

    def update_network_visualization(self, weights=None):
        if self.X_train is None or self.y_train is None:
            return
            
        hidden_sizes = []
        for layout in self.hidden_layer_widgets:
            spinbox = layout.itemAt(1).widget()
            if spinbox:
                hidden_sizes.append(spinbox.value())
        # Allow zero hidden layers - no default layer
        
        input_size = self.X_train.shape[1]
        output_size = len(torch.unique(self.y_train))
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # If no weights provided and model exists, get current weights
        if weights is None and self.model is not None:
            weights = self.model.state_dict()
        
        self.network_visualizer.visualize_network(layer_sizes, weights)

    def generate_custom_data(self):
        """
        This method is now deprecated. The logic has been moved to generate_data.
        It is kept to avoid breaking old connections but should be removed later.
        """
        # This function is now just a proxy to the main generate_data function
        # to ensure that if 'Custom' is not selected, it gets selected.
        for i in range(self.dataset_combo.count()):
            if self.dataset_combo.itemText(i) == "Custom":
                self.dataset_combo.setCurrentIndex(i)
                break
        # The generate_data function will be called automatically by the signal.

    def on_movement_mode_changed(self, state):
        # This method is called when the movement-based plotting checkbox state changes
        # You can add any necessary logic here
        pass

    def on_optimizer_changed(self):
        """Handle optimizer selection change to show/hide relevant parameters"""
        optimizer = self.optimizer_combo.currentText()
        
        # Show/hide momentum for SGD
        for i in range(self.momentum_layout.count()):
            item = self.momentum_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setVisible(optimizer == 'SGD')
        
        # Show/hide beta parameters for Adam-based optimizers
        for i in range(self.beta1_layout.count()):
            item = self.beta1_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setVisible(optimizer in ['Adam', 'AdamW'])
        
        for i in range(self.beta2_layout.count()):
            item = self.beta2_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setVisible(optimizer in ['Adam', 'AdamW'])

    def on_scheduler_changed(self):
        """Handle scheduler selection change to show/hide relevant parameters"""
        scheduler = self.scheduler_combo.currentText()
        
        # Show/hide scheduler parameters
        for i in range(self.scheduler_params_layout.count()):
            item = self.scheduler_params_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    widget.setVisible(scheduler != 'None')
        
        # Update scheduler parameter tooltip and range based on scheduler type
        if scheduler == 'StepLR':
            self.scheduler_param_spinbox.setToolTip("Step size for StepLR scheduler")
            self.scheduler_param_spinbox.setRange(1, 1000)
            self.scheduler_param_spinbox.setValue(30)
            self.scheduler_param_spinbox.setDecimals(0)
        elif scheduler == 'ExponentialLR':
            self.scheduler_param_spinbox.setToolTip("Gamma (multiplicative factor) for ExponentialLR")
            self.scheduler_param_spinbox.setRange(0.1, 2.0)
            self.scheduler_param_spinbox.setValue(0.95)
            self.scheduler_param_spinbox.setDecimals(3)
        elif scheduler == 'CosineAnnealingLR':
            self.scheduler_param_spinbox.setToolTip("T_max (period) for CosineAnnealingLR")
            self.scheduler_param_spinbox.setRange(1, 1000)
            self.scheduler_param_spinbox.setValue(100)
            self.scheduler_param_spinbox.setDecimals(0)

    def on_class_changed(self, index):
        """Called when the active class changes"""
        # Update the paint canvas title to reflect the new selected class
        if hasattr(self, 'paint_canvas'):
            self.paint_canvas.update_title()

    def add_class(self):
        """Add a new class to the system"""
        current_count = self.class_selector.count()
        if current_count < 10:  # Limit to 10 classes
            class_colors = ['Blue', 'Orange', 'Green', 'Red', 'Purple', 'Brown', 'Pink', 'Gray', 'Cyan', 'Yellow']
            new_class_name = f'Class {current_count} ({class_colors[current_count]})'
            self.class_selector.addItem(new_class_name)
            
            # Add the new class to the paint canvas
            if hasattr(self, 'paint_canvas'):
                self.paint_canvas.points[f'class_{current_count}'] = []
                self.paint_canvas.update_title()

    def remove_class(self):
        """Remove the last class from the system"""
        current_count = self.class_selector.count()
        if current_count > 2:  # Keep at least 2 classes
            # Remove the last item from the combo box
            self.class_selector.removeItem(current_count - 1)
            
            # Remove the last class from the paint canvas
            if hasattr(self, 'paint_canvas'):
                last_class_key = f'class_{current_count - 1}'
                if last_class_key in self.paint_canvas.points:
                    del self.paint_canvas.points[last_class_key]
                self.paint_canvas.update_title()

    def save_custom_data(self):
        """Save the custom-drawn data to a file."""
        X, y = self.paint_canvas.get_data()
        
        if X is None or y is None or len(X) == 0:
            QMessageBox.warning(self, "No Data", "There is no data to save.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Custom Data", "", "NumPy Data Files (*.npz)")
        
        if file_path:
            try:
                np.savez(file_path, X=X, y=y)
                QMessageBox.information(self, "Success", f"Data saved successfully to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save data: {e}")

    def load_custom_data(self):
        """Load custom data from a file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Custom Data", "", "NumPy Data Files (*.npz)")
        
        if file_path:
            try:
                data = np.load(file_path, allow_pickle=True)
                X = data['X']
                y = data['y']
                
                if X is not None and y is not None:
                    # Update class selector UI
                    unique_classes = np.unique(y)
                    self.class_selector.clear()
                    
                    class_colors = ['Blue', 'Orange', 'Green', 'Red', 'Purple', 'Brown', 'Pink', 'Gray', 'Cyan', 'Yellow']
                    for i in range(len(unique_classes)):
                        color_name = class_colors[i] if i < len(class_colors) else 'Unknown'
                        self.class_selector.addItem(f'Class {i} ({color_name})')

                    # Load data into paint canvas
                    self.paint_canvas.load_data(X, y)
                    QMessageBox.information(self, "Success", f"Data loaded successfully from {file_path}")
                    
                    # Automatically switch to the Custom dataset in the main tab
                    for i in range(self.dataset_combo.count()):
                        if self.dataset_combo.itemText(i) == "Custom":
                            self.dataset_combo.setCurrentIndex(i)
                            break
                else:
                    QMessageBox.warning(self, "Invalid File", "The selected file does not contain valid X and y data.")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {e}")

    def save_configuration(self):
        """Save the current architecture and learning parameters to a JSON file."""
        config = {
            'hidden_layers': [layout.itemAt(1).widget().value() for layout in self.hidden_layer_widgets],
            'learning_rate': self.lr_combo.currentText(),
            'activation': self.activation_combo.currentText(),
            'quantization': self.quantization_combo.currentText(),
            'optimizer': self.optimizer_combo.currentText(),
            'batch_size': self.batch_size_spinbox.value(),
            'weight_decay': self.weight_decay_spinbox.value(),
            'momentum': self.momentum_spinbox.value(),
            'beta1': self.beta1_spinbox.value(),
            'beta2': self.beta2_spinbox.value(),
            'scheduler': self.scheduler_combo.currentText(),
            'scheduler_param': self.scheduler_param_spinbox.value(),
            'early_stopping': self.early_stop_checkbox.isChecked(),
            'early_stop_patience': self.early_stop_patience_spinbox.value(),
            'dropout_rate': self.dropout_spinbox.value()
        }

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration", "", "JSON Files (*.json)")

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=4)
                QMessageBox.information(self, "Success", "Configuration saved successfully.")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save configuration: {e}")

    def load_configuration(self):
        """Load architecture and learning parameters from a JSON file."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Configuration", "", "JSON Files (*.json)")

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    config = json.load(f)

                # --- Restore Hidden Layers ---
                # Remove existing layers first, iterating backwards
                while self.hidden_layer_widgets:
                    self.remove_hidden_layer(self.hidden_layer_widgets[-1])
                
                # Add new layers from config
                for num_neurons in config.get('hidden_layers', []):
                    self.add_hidden_layer(neurons=num_neurons)

                # --- Restore Learning Parameters ---
                self.lr_combo.setCurrentText(config.get('learning_rate', '0.03'))
                self.activation_combo.setCurrentText(config.get('activation', 'ReLU'))
                self.quantization_combo.setCurrentText(config.get('quantization', '32-bit (None)'))
                self.optimizer_combo.setCurrentText(config.get('optimizer', 'Adam'))
                self.batch_size_spinbox.setValue(config.get('batch_size', 32))
                self.weight_decay_spinbox.setValue(config.get('weight_decay', 0.0))
                self.momentum_spinbox.setValue(config.get('momentum', 0.9))
                self.beta1_spinbox.setValue(config.get('beta1', 0.9))
                self.beta2_spinbox.setValue(config.get('beta2', 0.999))
                self.scheduler_combo.setCurrentText(config.get('scheduler', 'None'))
                self.scheduler_param_spinbox.setValue(config.get('scheduler_param', 0.1))
                self.early_stop_checkbox.setChecked(config.get('early_stopping', False))
                self.early_stop_patience_spinbox.setValue(config.get('early_stop_patience', 50))
                self.dropout_spinbox.setValue(config.get('dropout_rate', 0.0))

                # Update UI visibility for optimizer/scheduler specific params
                self.on_optimizer_changed()
                self.on_scheduler_changed()

                # Recreate the model with the new settings
                self.create_model()

                QMessageBox.information(self, "Success", "Configuration loaded successfully.")

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load configuration: {e}")

def main():
    app = QApplication(sys.argv)
    
    # Set a more professional font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    main_win = NeuralNetworkPlayground()
    main_win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 