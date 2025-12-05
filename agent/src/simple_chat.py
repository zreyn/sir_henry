from llama_cpp import Llama

llm = Llama(
    model_path="./models/Llama-3.2-3B-Q5_K_M.gguf",
    n_gpu_layers=-1,  # Offload all possible layers to the GPU
    verbose=True,  # Enable verbose output to see loading details
)
