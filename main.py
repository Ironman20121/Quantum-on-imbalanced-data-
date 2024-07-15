import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import EstimatorQNN

class ImageProcessingQiskit:
    objective_func_vals = []
    def __init__(self,training_path,testing_path) -> None:
        #trasnforming data 
        transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        # Load datasets using ImageFolder
        train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
        test_dataset = datasets.ImageFolder(test_data_path, transform=transform)
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        print(train_dataset.classes)
        # Convert train and test data to numpy arrays
        x_train, y_train = self.convert_to_numpy(train_loader)
        x_test, y_test = self.convert_to_numpy(test_loader)

        # Normalize the data
        x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
        x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

        # Ensure data dimensions are suitable for the quantum circuits 8 bit 
        x_train = x_train[:, :8]
        x_test = x_test[:, :8]
        
        
        
    ## Data perprocessing    
    # Function to convert image tensors to numpy arrays
    def convert_to_numpy(dataloader):
        images = []
        labels = []
        for data in dataloader:
            img, label = data
            img = img.numpy()
            img = img.reshape(img.shape[0], -1)  # Flatten the images
            images.append(img)
            labels.append(label)
        return np.concatenate(images), np.concatenate(labels)
    

    # Define convolutional layer
    def conv_layer(num_qubits, param_prefix):
        qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
        qubits = list(range(num_qubits))
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index : (param_index + 3)]), [q1, q2])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, qubits)
        return qc

    # Define pooling circuit
    def pool_circuit(params):
        target = QuantumCircuit(2)
        target.rz(-np.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        return target

    # Define pooling layer
    def pool_layer(sources, sinks, param_prefix):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling Layer")
        param_index = 0
        params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
        for source, sink in zip(sources, sinks):
            qc = qc.compose(pool_circuit(params[param_index : (param_index + 3)]), [source, sink])
            qc.barrier()
            param_index += 3

        qc_inst = qc.to_instruction()

        qc = QuantumCircuit(num_qubits)
        qc.append(qc_inst, range(num_qubits))
        return qc


    def callback_graph(weights, obj_func_eval):
        clear_output(wait=True)
        self.objective_func_vals.append(obj_func_eval)
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(self.objective_func_vals)), self.objective_func_vals)
        plt.show()

    
    