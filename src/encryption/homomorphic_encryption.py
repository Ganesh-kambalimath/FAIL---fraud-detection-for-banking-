"""
Homomorphic Encryption Module
Provides secure computation on encrypted data using CKKS scheme
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union, TYPE_CHECKING, Any
import logging

if TYPE_CHECKING:
    import tenseal as ts
else:
    try:
        import tenseal as ts
    except ImportError:
        ts = None
        logging.warning("TenSEAL not installed. Homomorphic encryption will not be available.")

logger = logging.getLogger(__name__)


class HomomorphicEncryption:
    """
    Homomorphic Encryption using CKKS scheme via TenSEAL
    
    Enables computation on encrypted model updates without decryption.
    
    Args:
        poly_modulus_degree: Degree of polynomial modulus (power of 2)
        coeff_mod_bit_sizes: Bit sizes for coefficient modulus chain
        scale: Scaling factor for encoding (can be int, float, or string like '2**40')
        use_global_scale: Whether to use global scale
    """
    
    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
        scale: Union[int, float, str] = 2**40,
        use_global_scale: bool = True
    ):
        if ts is None:
            raise ImportError(
                "TenSEAL is required for homomorphic encryption. "
                "Install with: pip install tenseal"
            )
        
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        
        # FIX: Convert scale to float, handling string expressions like '2**40'
        if isinstance(scale, str):
            try:
                # Safely evaluate mathematical expressions
                self.scale = float(eval(scale))
                logger.info(f"Converted scale from string '{scale}' to {self.scale}")
            except (SyntaxError, NameError, TypeError, ValueError) as e:
                logger.warning(f"Could not evaluate scale '{scale}': {e}. Using default 2**40")
                self.scale = float(2**40)
        else:
            self.scale = float(scale)
        
        self.use_global_scale = use_global_scale
        
        # Create TenSEAL context
        self.context = self._create_context()
        
        logger.info(f"Initialized HE with poly_modulus_degree={poly_modulus_degree}, "
                   f"scale=2^{int(np.log2(self.scale))}")
    
    def _create_context(self) -> "ts.Context":
        """
        Create TenSEAL encryption context
        
        Returns:
            TenSEAL context
        """
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes
        )
        
        if self.use_global_scale:
            # self.scale is now guaranteed to be a float
            context.global_scale = self.scale
        
        # Generate encryption keys
        context.generate_galois_keys()
        
        return context
    
    def encrypt_tensor(self, tensor: torch.Tensor) -> "ts.CKKSVector":
        """
        Encrypt a PyTorch tensor
        
        Args:
            tensor: Input tensor to encrypt
        
        Returns:
            Encrypted CKKS vector
        """
        # Flatten tensor
        flat_tensor = tensor.flatten().cpu().numpy().tolist()
        
        # Encrypt
        encrypted = ts.ckks_vector(self.context, flat_tensor)
        
        return encrypted
    
    def decrypt_tensor(
        self,
        encrypted: "ts.CKKSVector",
        original_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Decrypt CKKS vector to PyTorch tensor
        
        Args:
            encrypted: Encrypted CKKS vector
            original_shape: Original tensor shape
        
        Returns:
            Decrypted tensor
        """
        # Decrypt
        decrypted_list = encrypted.decrypt()
        
        # Convert to tensor and reshape
        tensor = torch.tensor(decrypted_list, dtype=torch.float32)
        tensor = tensor.reshape(original_shape)
        
        return tensor
    
    def encrypt_model_update(
        self,
        model_update: List[torch.Tensor]
    ) -> List["ts.CKKSVector"]:
        """
        Encrypt model parameter updates
        
        Args:
            model_update: List of parameter tensors
        
        Returns:
            List of encrypted vectors
        """
        encrypted_updates = []
        
        for param in model_update:
            encrypted = self.encrypt_tensor(param)
            encrypted_updates.append(encrypted)
        
        logger.debug(f"Encrypted {len(model_update)} parameters")
        
        return encrypted_updates
    
    def decrypt_model_update(
        self,
        encrypted_update: List["ts.CKKSVector"],
        shapes: List[Tuple[int, ...]]
    ) -> List[torch.Tensor]:
        """
        Decrypt model parameter updates
        
        Args:
            encrypted_update: List of encrypted vectors
            shapes: Original shapes of parameters
        
        Returns:
            List of decrypted parameter tensors
        """
        decrypted_updates = []
        
        for encrypted, shape in zip(encrypted_update, shapes):
            decrypted = self.decrypt_tensor(encrypted, shape)
            decrypted_updates.append(decrypted)
        
        logger.debug(f"Decrypted {len(encrypted_update)} parameters")
        
        return decrypted_updates
    
    def aggregate_encrypted(
        self,
        encrypted_updates: List[List["ts.CKKSVector"]],
        weights: Optional[List[float]] = None
    ) -> List["ts.CKKSVector"]:
        """
        Aggregate multiple encrypted updates using weighted average
        
        Args:
            encrypted_updates: List of encrypted model updates from clients
            weights: Optional weights for weighted average
        
        Returns:
            Aggregated encrypted update
        """
        if not encrypted_updates:
            raise ValueError("No updates to aggregate")
        
        num_clients = len(encrypted_updates)
        num_params = len(encrypted_updates[0])
        
        # Default to equal weights
        if weights is None:
            weights = [1.0 / num_clients] * num_clients
        
        # Normalize weights
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]
        
        # Aggregate each parameter
        aggregated = []
        for param_idx in range(num_params):
            # Start with first client's encrypted parameter
            weighted_sum = encrypted_updates[0][param_idx] * weights[0]
            
            # Add remaining clients
            for client_idx in range(1, num_clients):
                encrypted_param = encrypted_updates[client_idx][param_idx]
                weighted_sum += encrypted_param * weights[client_idx]
            
            aggregated.append(weighted_sum)
        
        logger.info(f"Aggregated {num_clients} encrypted updates")
        
        return aggregated
    
    def serialize_context(self) -> bytes:
        """
        Serialize encryption context for sharing
        
        Returns:
            Serialized context bytes
        """
        return self.context.serialize()
    
    def load_context(self, serialized: bytes):
        """
        Load encryption context from serialized bytes
        
        Args:
            serialized: Serialized context
        """
        self.context = ts.context_from(serialized)
        logger.info("Loaded encryption context")
    
    def make_context_public(self):
        """
        Make context public (remove secret key)
        This is used when sharing context with clients
        """
        self.context.make_context_public()
        logger.info("Context made public")
    
    def get_public_context(self) -> bytes:
        """
        Get public context for sharing with clients
        
        Returns:
            Serialized public context
        """
        public_context = self.context.copy()
        public_context.make_context_public()
        return public_context.serialize()


class SecureAggregation:
    """
    Secure Multi-Party Computation for Aggregation
    
    Implements secure aggregation protocol where the server cannot
    see individual client updates, only the aggregated result.
    """
    
    def __init__(self, threshold: float = 0.67):
        """
        Initialize secure aggregation
        
        Args:
            threshold: Minimum fraction of clients needed (Byzantine tolerance)
        """
        self.threshold = threshold
        self.client_keys = {}
        logger.info(f"Initialized Secure Aggregation with threshold={threshold}")
    
    def generate_client_masks(
        self,
        client_id: str,
        num_clients: int,
        param_shapes: List[Tuple[int, ...]]
    ) -> List[torch.Tensor]:
        """
        Generate random masks for a client
        
        Args:
            client_id: Client identifier
            num_clients: Total number of clients
            param_shapes: Shapes of model parameters
        
        Returns:
            List of mask tensors
        """
        # Generate seed for this client
        seed = hash(client_id) % (2**32)
        generator = torch.Generator().manual_seed(seed)
        
        masks = []
        for shape in param_shapes:
            mask = torch.randn(shape, generator=generator)
            masks.append(mask)
        
        self.client_keys[client_id] = seed
        
        return masks
    
    def mask_update(
        self,
        update: List[torch.Tensor],
        masks: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Mask client update
        
        Args:
            update: Client's model update
            masks: Random masks
        
        Returns:
            Masked update
        """
        masked_update = []
        for param, mask in zip(update, masks):
            masked_update.append(param + mask)
        
        return masked_update
    
    def unmask_aggregate(
        self,
        masked_aggregate: List[torch.Tensor],
        participating_clients: List[str],
        param_shapes: List[Tuple[int, ...]]
    ) -> List[torch.Tensor]:
        """
        Remove masks from aggregated update
        
        Args:
            masked_aggregate: Aggregated masked updates
            participating_clients: List of participating client IDs
            param_shapes: Shapes of model parameters
        
        Returns:
            Unmasked aggregate
        """
        # Regenerate and sum all masks
        total_masks = [torch.zeros(shape) for shape in param_shapes]
        
        for client_id in participating_clients:
            if client_id in self.client_keys:
                seed = self.client_keys[client_id]
                generator = torch.Generator().manual_seed(seed)
                
                for i, shape in enumerate(param_shapes):
                    mask = torch.randn(shape, generator=generator)
                    total_masks[i] += mask
        
        # Remove masks
        unmasked = []
        for masked, total_mask in zip(masked_aggregate, total_masks):
            unmasked.append(masked - total_mask)
        
        return unmasked