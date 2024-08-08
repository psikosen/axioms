import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Object Representation
class Object3D:
    def __init__(self, x, y, z):
        self.coordinates = np.array([x, y, z])

    def communicate(self, other_object):
        # Example communication: averaging the coordinates
        self.coordinates = (self.coordinates + other_object.coordinates) / 2
        other_object.coordinates = self.coordinates

# Complex Neural Network
class ComplexNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(ComplexNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Rotation Node
class RotationNode:
    def __init__(self, rotation_matrix, neural_net):
        self.rotation_matrix = rotation_matrix
        self.neural_net = neural_net
        self.neighbors = []
        self.objects = []

    def add_neighbor(self, neighbor_node):
        self.neighbors.append(neighbor_node)

    def add_object(self, obj):
        self.objects.append(obj)

    def rotate_objects(self):
        for obj in self.objects:
            # Apply rotation
            obj.coordinates = self.rotation_matrix.dot(obj.coordinates)
            # Neural net processing
            input_tensor = torch.tensor(obj.coordinates, dtype=torch.float32)
            obj.coordinates = self.neural_net(input_tensor).detach().numpy()

    def communicate_objects(self):
        if len(self.objects) > 1:
            for i in range(len(self.objects) - 1):
                self.objects[i].communicate(self.objects[i + 1])

# Rotation Graph
class RotationGraph:
    def __init__(self):
        self.nodes = []

    def add_node(self, rotation_matrix, neural_net):
        node = RotationNode(rotation_matrix, neural_net)
        self.nodes.append(node)
        return node

    def add_object_to_node(self, node, obj):
        node.add_object(obj)

    def rotate_all_objects(self):
        for node in self.nodes:
            node.rotate_objects()
            node.communicate_objects()
            # Exchange information with neighbors
            for neighbor in node.neighbors:
                self.exchange_information(node, neighbor)

    def exchange_information(self, node, neighbor):
        # Placeholder for information exchange logic
        pass

# Utility function for creating rotation matrices
def create_rotation_matrix(angle, axis):
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

# Generate synthetic data for training
def generate_synthetic_data():
    np.random.seed(42)
    X = np.random.rand(1000, 3) * 10  # Random points in 3D space
    y = np.random.rand(1000, 3) * 10  # Random target points in 3D space
    return X, y

# Main execution logic for training and testing with synthetic data
X, y = generate_synthetic_data()

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the complex neural network
model = ComplexNN(3, 3)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Testing the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor).item()

print(f'Test Loss: {test_loss:.4f}')

# Print sample predictions
print("Sample Predictions:")
print(predictions[:5])
print("Actual Values:")
print(y_test_tensor[:5])

# Main execution logic for rotation graph
graph = RotationGraph()

# Define rotation matrices
rot_x = create_rotation_matrix(np.pi / 4, 'x')
rot_y = create_rotation_matrix(np.pi / 4, 'y')
rot_z = create_rotation_matrix(np.pi / 4, 'z')

# Add nodes to the graph
node_x = graph.add_node(rot_x, model)
node_y = graph.add_node(rot_y, model)
node_z = graph.add_node(rot_z, model)

# Add neighbors
node_x.add_neighbor(node_y)
node_y.add_neighbor(node_z)

# Create objects
obj1 = Object3D(1, 1, 1)
obj2 = Object3D(2, 2, 2)
obj3 = Object3D(3, 3, 3)
obj4 = Object3D(4, 4, 4)

# Add objects to nodes
graph.add_object_to_node(node_x, obj1)
graph.add_object_to_node(node_x, obj2)
graph.add_object_to_node(node_y, obj3)
graph.add_object_to_node(node_y, obj4)

# Perform rotation and information exchange
#graph.rotate_all_objects()

g = 0
for g in range(6):
 graph.rotate_all_objects()

# Print coordinates of objects after rotation and communication
results = []
for node in graph.nodes:
    for obj in node.objects:
        results.append(obj.coordinates)

print("Coordinates after rotation and communication:")
for coords in results:
    print(coords)

# Now, let's test the neural network against the Breast Cancer dataset

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Convert targets to one-hot encoding
y = np.eye(2)[y]

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Define the neural network for Breast Cancer dataset
class CancerNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CancerNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, output_size)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

cancer_model = CancerNN(30, 2)

# Define loss function and optimizer for Breast Cancer dataset
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cancer_model.parameters(), lr=0.001)

# Training loop for Breast Cancer dataset
epochs = 100
for epoch in range(epochs):
    cancer_model.train()
    optimizer.zero_grad()
    outputs = cancer_model(X_train_tensor)
    loss = criterion(outputs, torch.argmax(y_train_tensor, dim=1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Testing the Breast Cancer model
cancer_model.eval()
with torch.no_grad():
    predictions = cancer_model(X_test_tensor)
    test_loss = criterion(predictions, torch.argmax(y_test_tensor, dim=1)).item()

# Calculate accuracy
_, predicted_labels = torch.max(predictions, 1)
_, actual_labels = torch.max(y_test_tensor, 1)
accuracy = (predicted_labels == actual_labels).float().mean().item()

print(f'Test Loss: {test_loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# Print sample predictions
print("Sample Predictions:")
print(predicted_labels[:5])
print("Actual Values:")
print(actual_labels[:5])



