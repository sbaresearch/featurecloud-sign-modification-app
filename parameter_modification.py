import numpy as np
import onnx
import struct
from onnx import numpy_helper
import numpy as np




def flip_lowest_values_based_on_threshold(param_array, percentage=0.1):
    """
    Flip the percentage of lowest absolute values in the array to a random small positive or negative value,
    where the range is determined by the threshold calculated from the percentage of the lowest values.
    """
    # Flatten the array to work with it as a 1D array
    flat_array = param_array.flatten()
    # Calculate the number of values to flip
    num_values_to_flip = int(len(flat_array) * percentage)
    # Find the threshold below which values will be flipped
    threshold = np.partition(abs(flat_array), num_values_to_flip)[num_values_to_flip]
    # Identify indices below the threshold for negative and positive values separately
    negative_indices = (flat_array < 0) & (abs(flat_array) < threshold)
    positive_indices = (flat_array >= 0) & (abs(flat_array) < threshold)
    # Generate random positive small values within the range [0, threshold] for negative values
    random_positive_values = np.random.uniform(0, threshold, size=np.sum(negative_indices))
    # Generate random negative small values within the range [-threshold, 0] for positive values
    random_negative_values = np.random.uniform(-threshold, 0, size=np.sum(positive_indices))
    # Apply the generated random values separately
    flat_array[negative_indices] = random_positive_values
    flat_array[positive_indices] = random_negative_values
    # Reshape the array back to its original shape
    return flat_array.reshape(param_array.shape)

def modify_and_update_network(model, percentage_to_modify):
    for initializer in model.graph.initializer:
        # Check if the data is in raw format
        if initializer.HasField('raw_data'):
            weight_data = numpy_helper.to_array(initializer)
        else:
            # Assume the data is in float format
            weight_data = np.array(initializer.float_data).reshape(initializer.dims)

        if percentage_to_modify > 0:
            # Flatten the weight data for easier processing
            flattened_weights = weight_data.flatten()
            # Flip the values (i.e. signs are flipped too) of the values below threshold given by %
            modified_weight_data = flip_lowest_values_based_on_threshold(weight_data, percentage_to_modify)
            # Create a new initializer with the modified array
            new_initializer = numpy_helper.from_array(modified_weight_data, name=initializer.name)
            # Replace the old initializer with the modified one
            initializer.CopyFrom(new_initializer)

            # Reshape the modified weights to the original shape
            modified_weights = np.array(weight_data).reshape(weight_data.shape)

            # Convert the modified weights to the serialized binary format
            serialized_data = modified_weights.astype(np.float32).tobytes()

            # Clear the existing data fields
            initializer.ClearField('float_data')

            # Update the raw_data field of the initializer
            initializer.raw_data = serialized_data

    return model

# Function to prune the connections from the network
def modify_signs(model, percentage_to_modify):
    modified_model = modify_and_update_network(model, percentage_to_modify)
    return modified_model



def compare_original_to_modified(original_model, modified_model):
    # Load the original and modified ONNX models

    # Extract initializers (parameters) from both models
    original_initializers = {init.name: numpy_helper.to_array(init) for init in original_model.graph.initializer}
    modified_initializers = {init.name: numpy_helper.to_array(init) for init in modified_model.graph.initializer}

    # Check if the initializers in both models have the same structure
    same_structure = True
    if set(original_initializers.keys()) != set(modified_initializers.keys()):
        same_structure = False
    else:
        for key in original_initializers:
            if original_initializers[key].shape != modified_initializers[key].shape:
                same_structure = False
                break

    return same_structure


def compare_graph_structure(original_model, modified_model):
    # Check if the nodes in both graphs have the same structure
    if len(original_model.graph.node) != len(modified_model.graph.node):
        return False

    for original_node, modified_node in zip(original_model.graph.node, modified_model.graph.node):
        if original_node.op_type != modified_node.op_type:
            return False
        if original_node.input != modified_node.input:
            return False
        if original_node.output != modified_node.output:
            return False

    return True
