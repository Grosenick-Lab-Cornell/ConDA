import sklearn.linear_model
from cebra import CEBRA

N = 2

# Define a CEBRA model
cebra_model = CEBRA(
    model_architecture="offset10-model-mse", #consider: "offset10-model-mse" if Euclidean
    batch_size=512,
    learning_rate=3e-4,
    #temperature_mode='constant',
    temperature=1.1,
    #delta=0.01,
    max_iterations=5000, #we will sweep later; start with default
    conditional='time', #for supervised, put 'time_delta', or 'delta'
    output_dimension=N,
    #distance='euclidean', #consider 'euclidean'; if you set this, output_dimension min=2
    device="cuda_if_available",
    #num_hidden_units = 32,
    verbose=True,
    time_offsets=15
)
