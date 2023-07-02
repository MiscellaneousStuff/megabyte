def calculate_transformer_params(V, H, A, N, D_ff):
    """
    V: Vocabulary size
    H: Hidden size (embedding size)
    A: Number of attention heads
    N: Number of layers (transformer blocks)
    D_ff: Feed-forward inner layer dimension (usually 4*H)
    """
    # Input Embeddings
    input_embeddings = V * H

    # Transformer layers
    transformer_layers = N * ((3 * H * A) + H + (2 * H * D_ff))

    # Output Layer
    # Assuming output softmax layer shares weights with the input embeddings
    output_layer = 0 

    total_params = input_embeddings + transformer_layers + output_layer

    return total_params

# Example usage
V = 50257  # Example: Vocabulary size for GPT-2
H = 768  # Example: Hidden size for GPT-2 base model
A = 12  # Example: Attention heads for GPT-2 base model
N = 12  # Example: Layers for GPT-2 base model
D_ff = 4 * H  # Typically D_ff is 4*H

print("Total parameters: ", calculate_transformer_params(V, H, A, N, D_ff))