# Neural Network Playground

An interactive neural network simulation playground built with Python and PyQt6. This application provides a user-friendly interface for experimenting with neural networks, visualizing their architecture, and training them on various datasets.

## Features

- **Interactive Network Visualization**: Real-time visualization of neural network architecture
- **Multiple Datasets**: Support for Moons, Circles, and Classification datasets
- **Customizable Architecture**: Adjustable input size, hidden layers, and output size
- **Multiple Activation Functions**: ReLU, Sigmoid, and Tanh activation functions
- **Real-time Training**: Live training progress with loss and accuracy metrics
- **Data Visualization**: Interactive plots of training data and decision boundaries
- **Modern Dark UI**: Clean and intuitive user interface

## Quick Start

### 1. Activate the Virtual Environment

**Windows:**
```bash
# Option 1: Use the batch file
.\activate_env.bat

# Option 2: Manual activation
neural_env\Scripts\activate
```

**Linux/Mac:**
```bash
source neural_env/bin/activate
```

### 2. Run the Application

```bash
# Main GUI application
python neural_network_playground.py

# Or use the smart launcher
python run_playground.py

## Installation (If Starting Fresh)

1. **Create virtual environment:**
   ```bash
   python -m venv neural_env
   ```

2. **Activate it:**
   ```bash
   # Windows
   neural_env\Scripts\activate
   
   # Linux/Mac
   source neural_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install PyQt6 numpy matplotlib scikit-learn torch networkx
   ```

## Usage

1. **Generate Data**: Choose a dataset type (Moons, Circles, or Classification) and click "Generate Data"
2. **Create Model**: Adjust the network architecture parameters and click "Create Model"
3. **Train**: Set training parameters and click "Start Training" to begin training
4. **Monitor**: Watch the training progress in real-time and view the network architecture

## Controls

### Network Architecture
- **Input Size**: Number of input features
- **Hidden Layers**: Number of hidden layers
- **Hidden Layer Size**: Number of neurons in each hidden layer
- **Output Size**: Number of output classes
- **Activation Function**: Choose between ReLU, Sigmoid, and Tanh

### Training Parameters
- **Learning Rate**: Adjust the learning rate using the slider (0.001 to 0.1)
- **Epochs**: Number of training epochs

### Data Generation
- **Dataset Type**: Choose from Moons, Circles, or Classification datasets
- **Samples**: Number of training samples to generate

## Features in Detail

### Network Visualization
The application provides an interactive visualization of the neural network architecture, showing:
- Nodes representing neurons
- Connections between layers
- Layer sizes and structure

### Data Visualization
- 2D scatter plots of training data
- Color-coded by class labels
- Decision boundary visualization (when available)

### Training Progress
- Real-time loss and accuracy updates
- Training log with detailed information
- Ability to stop training at any time

## Technical Details

- **Framework**: PyQt6 for the GUI
- **Neural Network**: PyTorch for model implementation
- **Visualization**: Matplotlib for plots and NetworkX for network graphs
- **Data**: Scikit-learn for dataset generation

## Requirements

- Python 3.8+
- PyQt6
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- NetworkX

## Troubleshooting

### Common Issues:

1. **Import errors**: Make sure the virtual environment is activated
   ```bash
   # Check if environment is active (should show neural_env)
   echo $env:VIRTUAL_ENV  # Windows PowerShell
   ```

2. **GUI not showing**: Check if PyQt6 is properly installed
   ```bash
   pip list | findstr PyQt6
   ```

3. **Matplotlib backend errors**: The application uses the Qt backend for PyQt6

4. **Training errors**: Verify PyTorch installation
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

### Environment Commands:

```bash
# Deactivate environment
deactivate

# Check installed packages
pip list

# Update packages
pip install --upgrade package_name
```

## Project Structure

```
neural_networks/
├── neural_env/                    # Virtual environment
├── neural_network_playground.py   # Main application
├── run_playground.py             # Smart launcher
├── demo.py                       # Command-line demo
├── test_installation.py          # Installation test
├── requirements.txt              # Dependencies
├── activate_env.bat              # Environment activator
├── README.md                     # This file
└── SETUP.md                      # Detailed setup guide
```

## Contributing

Feel free to contribute to this project by:
- Adding new features
- Improving the UI
- Adding more dataset types
- Enhancing the visualization capabilities

## License

This project is open source and available under the MIT License. 