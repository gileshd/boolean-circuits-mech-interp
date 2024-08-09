from flax import linen as nn
from jax import numpy as jnp
from jaxtyping import Array, Float


class Encoder(nn.Module):
    """Encoder with a single hidden layer and ReLU activation."""
    hidden_dim: int

    @nn.compact
    def __call__(
        self, inputs: Float[Array, "*batch data_dim"]
    ) -> Float[Array, "*batch hidden_dim"]:
        x = nn.Dense(self.hidden_dim, use_bias=False)(inputs)
        x = nn.relu(x)
        return x

class Decoder(nn.Module):
    """Decoder with a single hidden layer and ReLU activation."""
    output_dim: int

    # @nn.compact
    # def __call__(
    #     self, inputs: Float[Array, "*batch hidden_dim"]
    # ) -> Float[Array, "*batch output_dim"]:
    #     x = nn.Dense(self.output_dim, use_bias=False)(inputs)
    #     return x
    @nn.compact
    def __call__(self, x):
        # Assuming the first layer of the decoder is a Dense layer
        # Retrieve the weight and bias
        kernel = self.param('kernel', nn.initializers.lecun_normal(), (x.shape[-1], self.output_dim))
        
        # Normalize the kernel to have unit norm
        kernel_normalized = kernel / jnp.linalg.norm(kernel, axis=0, keepdims=True)
        
        # Apply the modified dense layer operation
        x = jnp.dot(x, kernel_normalized)
        return x

class AutoEncoder(nn.Module):
    """Auto-encoder with a single hidden layer and ReLU activation."""
    input_dim: int
    hidden_dim: int

    def setup(self):
        self.encoder = Encoder(hidden_dim=self.hidden_dim)
        self.decoder = Decoder(output_dim=self.input_dim)

    def __call__(
        self, inputs: Float[Array, "*batch data_dim"]
    ) -> Float[Array, "*batch data_dim"]:
        hidden = self.encoder(inputs)
        recon = self.decoder(hidden)
        return recon

    def encode(self, inputs: Float[Array, "*batch data_dim"]) -> Float[Array, "*batch hidden_dim"]:
        return self.encoder(inputs)

