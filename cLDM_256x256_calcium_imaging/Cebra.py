import sklearn.linear_model
from cebra import CEBRA

cebra_model = CEBRA(
    model_architecture='offset10-model-mse',
    batch_size=512,
    learning_rate=5e-6,
    output_dimension=3,
    delta=20,
    temperature=1.12,
    max_iterations=5000,
    distance='euclidean',
    conditional='time',
    device='cuda_if_available',
    verbose=True,
    time_offsets=8
)