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
    elif source.ndim == 4:

        #
        if destination.ndim == 4:
            #
            destination[:source.shape[0], :source.shape[1], :source.shape[2], :source.shape[3]] = source

    #
    return destination


#
###
#
def create_permission_vector(nb_permissions_items: int, permission_item: int, dtype: torch.dtype, device: str | torch.device) -> Tensor:

    #
    v: Tensor = torch.zeros( (1, nb_permissions_items), dtype=dtype, device=device )
    v[0, permission_item] = 1

    #
    return v


#
### Chunk Class. ###
#
class Chunk:

    #
    def __init__(
        self,
        permissions_items: dict[str, int],
        chunk_length: int = 512,
        chunk_global_context_length: int = 5,
        initial_data: Optional[Tensor] = None,
        initial_data_permissions_mask: Optional[Tensor] = None,
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
        self.permissions_items: dict[str, int] = permissions_items
        self.nb_permissions_items: int = len(permissions_items)

        #
        self.chunk_length: int = chunk_length
        self.chunk_global_context_length: int = chunk_global_context_length

        #
        ### Prepare Tensor Shapes ###
        #
        self.chunk_context_shape: tuple[int, ...] = (self.chunk_length,)
        self.permissions_mask_context_shape: tuple[int, ...] = (self.chunk_length, self.nb_permissions_items)
        #
        self.chunk_global_context_shape: tuple[int, ...] = (self.chunk_global_context_length,)
        self.permissions_mask_global_context_shape: tuple[int, ...] = (self.chunk_global_context_length, self.nb_permissions_items)

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
        ### Create chunk global context data. ###
        #
        self.chunk_global_context_data: Tensor = torch.full(
            size=self.chunk_global_context_shape,
            fill_value=self.padding_token,
            dtype=torch.int64,
            device=self.device
        )
        #
        ### Create permissions mask for chunk global context data. ###
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
        permission_hidden: Tensor = create_permission_vector(nb_permissions_items = self.nb_permissions_items, permission_item = self.permissions_items["hidden"], dtype = self.dtype, device = self.device)
        permission_read_and_write_inside_chunk: Tensor = create_permission_vector(nb_permissions_items = self.nb_permissions_items, permission_item = self.permissions_items["chunk_inside_read_and_write"], dtype = self.dtype, device = self.device)
        permission_read_and_write_global_chunk: Tensor = create_permission_vector(nb_permissions_items = self.nb_permissions_items, permission_item = self.permissions_items["chunk_global_read_and_write"], dtype = self.dtype, device = self.device)
        #
        idx_permissions_hidden: Tensor = self.chunk_context_data[:] == self.padding_token
        #
        ## `~` is the logical `not` operator. ##
        #
        idx_permissions_normal: Tensor = ( ~idx_permissions_hidden )
        #
        self.permission_mask_context_data[idx_permissions_hidden] = permission_hidden
        self.permission_mask_context_data[idx_permissions_normal] = permission_read_and_write_inside_chunk
        #
        self.permission_mask_global_context_data[:] = permission_read_and_write_global_chunk

        #
        ### Fill permissions mask for context data with initial data if provided. ###
        #
        if initial_data_permissions_mask is not None:
            #
            self.permission_mask_context_data = set_matrix_initial_data(destination=self.permission_mask_context_data, source=initial_data_permissions_mask)
