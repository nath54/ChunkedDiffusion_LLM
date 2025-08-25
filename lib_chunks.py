#
### Import Modules. ###
#
from typing import Optional
#
import torch
from torch import Tensor
#
from lib_get_device import get_best_device


#
def set_matrix_initial_data(destination: Tensor, source: Tensor) -> Tensor:

    #
    if source.ndim == 1:

        #
        if destination.ndim == 1:
            #
            destination[:source.shape[0]] = source

        #
        elif destination.ndim == 2:
            #
            destination[0, :source.shape[0]] = source

        #
        elif destination.ndim == 3:
            #
            destination[0, 0, :source.shape[0]] = source

        #
        elif destination.ndim == 4:
            #
            destination[0, 0, 0, :source.shape[0]] = source

    #
    elif source.ndim == 2:

        #
        if destination.ndim == 2:
            #
            destination[:source.shape[0], :source.shape[1]] = source

        #
        elif destination.ndim == 3:
            #
            destination[0, :source.shape[0], :source.shape[1]] = source

        #
        elif destination.ndim == 4:
            #
            destination[0, 0, :source.shape[0], :source.shape[1]] = source

    #
    elif source.ndim == 3:

        #
        if destination.ndim == 3:
            #
            destination[:source.shape[0], :source.shape[1], :source.shape[2]] = source

        #
        elif destination.ndim == 4:
            #
            destination[0, :source.shape[0], :source.shape[1], :source.shape[2]] = source

    #
    elif source.ndim == 3:

        #
        if destination.ndim == 4:
            #
            destination[:source.shape[0], :source.shape[1], :source.shape[2], :source.shape[3]] = source

    #
    return destination


#
### Chunk Class. ###
#
class Chunk:

    #
    def __init__(
        self,
        chunk_length: int = 512,
        chunk_global_context_length: int = 5,
        batch_size: Optional[int] = None,
        initial_data: Optional[Tensor] = None,
        initial_data_permissions_mask: Optional[Tensor] = None,
        nb_permissions_items: int = 9,
        padding_token: int = 0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = get_best_device()
    ) -> None:

        #
        self. dtype: torch.dtype = dtype
        self. device: str | torch.device = device
        #
        self.padding_token: int = padding_token
        #
        self.nb_permissions_items: int = nb_permissions_items

        #
        self.chunk_length: int = chunk_length
        self.chunk_global_context_length: int = chunk_global_context_length

        #
        ### Prepare Tensor Shapes ###
        #
        self.chunk_context_shape: tuple[int, ...] = (self.chunk_length,)
        self.permissions_mask_context_shape: tuple[int, ...] = (self.chunk_length, nb_permissions_items)
        #
        self.chunk_global_context_shape: tuple[int, ...] = (self.chunk_global_context_length,)
        self.permissions_mask_global_context_shape: tuple[int, ...] = (self.chunk_global_context_length, nb_permissions_items)
        #
        if batch_size is not None:
            #
            self.chunk_context_shape = (batch_size, chunk_length,)
            self.permissions_mask_context_shape = (batch_size, chunk_length, nb_permissions_items)
            #
            self.chunk_global_context_shape = (batch_size, self.chunk_global_context_length,)
            self.permissions_mask_global_context_shape = (batch_size, self.chunk_global_context_length, nb_permissions_items)

        #
        ### Create chunk context data. ###
        #
        self.chunk_context_data: Tensor = torch.full(
            size=self.chunk_context_shape,
            fill_value=self.padding_token,
            dtype=torch.int64,
            device=self.device
        )
        #
        ### Fill chunk context data with initial data if provided. ###
        #
        if initial_data is not None:
            #
            self.chunk_context_data = set_matrix_initial_data(destination=self.chunk_context_data, source=initial_data)

        #
        ### Create permissions mask for chunk context data. ###
        #
        self.permission_mask_context_data: Tensor = torch.full(
            size=self.permissions_mask_context_shape,
            fill_value=0,
            dtype=self.dtype,
            device=self.device
        )
        #
        self.permission_mask_context_data
        #
        ### Fill permissions mask for context data with initial data if provided. ###
        #
        if initial_data_permissions_mask is not None:
            #
            self.permission_mask_context_data = set_matrix_initial_data(destination=self.chunk_context_data, source=initial_data_permissions_mask)

        #
        ### Create chunk global context data. ###
        #
        self.chunk_global_context_data: Tensor = torch.full(
            size=self.chunk_global_context_shape,
            fill_value=self.padding_token,
            dtype=torch.int64,
            device=self.device
        )
        #
        ### Create permissions mask for chunk context data. ###
        #
        self.permission_mask_global_context_data: Tensor = torch.full(
            size=self.permissions_mask_global_context_shape,
            fill_value=0,
            dtype=self.dtype,
            device=self.device
        )

        #
        ### Set padding tokens hidden. ###
        #
        permission_hidden: Tensor = Tensor([1] + [0] * self.nb_permissions_items)
        #
        # TODO: set padding tokens hidden.
