import torch
from typing import Optional, Tuple
from .dot_product_attention import TorchMultiStageDotProductAttention
import functools
import random
import os, psutil

class CudaCache:
    def __init__(self, num_units, unit_size, max_block_size, dtype, device='cuda'):
        self.num_units = num_units
        self.unit_size = unit_size
        self.dtype = dtype
        self.data = torch.empty(
            (num_units, unit_size),
            device=device,
            dtype=dtype
        )
        self.idle_set = set(list(range(num_units)))
        self.max_block_size = max_block_size

    def alloc(self):
        assert len(self.idle_set) > 0, "No more idle units in cache."
        idx = self.idle_set.pop()
        return self.data[idx], idx

    def delete(self, idx):
        assert idx not in self.idle_set
        self.idle_set.add(idx)


class MemoryBlock:
    _instance_counter = 0

    def __init__(
        self, 
        kv: Tuple[torch.Tensor, torch.Tensor], 
        cache: CudaCache, 
        load_to_cache: bool = False, 
        pin_memory: bool = False,
        allow_disk_offload: bool = False,
        offload_dir: str = "./offload_data",
        load_to_disk: bool = False,
        cpu_cache: Optional[CudaCache] = None
    ):
        if allow_disk_offload is not False:
            # set up disk offloading
            self.allow_disk_offload = True
            self.id = MemoryBlock._instance_counter
            MemoryBlock._instance_counter += 1
            self.offload_dir = offload_dir + f"/{self.id // 10000}"  # reduce number of files per directory to 10,000 max.
            os.makedirs(self.offload_dir + '/0', exist_ok=True)
            os.makedirs(self.offload_dir + '/1', exist_ok=True)
        else:
            self.allow_disk_offload = False
    
        num_heads_kv, size, dim_head = kv[0].shape
        self.cache = cache
        self.cpu_cache = cpu_cache
        self.on_disk = False
        assert size <= self.cache.max_block_size 

        if load_to_disk:
            torch.save(kv[0].contiguous(), os.path.join(self.offload_dir, f"0/{self.id}.pt"), pickle_protocol=4)
            torch.save(kv[1].contiguous(), os.path.join(self.offload_dir, f"1/{self.id}.pt"), pickle_protocol=4)
            self.on_disk = True
            cpu_data = None
            self.cpu_data_id = None
        elif cpu_cache is not None:
            cpu_data, cpu_data_id = cpu_cache.alloc()
            cpu_data = cpu_data.view((2, num_heads_kv, self.cache.max_block_size, dim_head))
            cpu_data[0][:, :size, :].copy_(kv[0].contiguous(), non_blocking=True)
            cpu_data[1][:, :size, :].copy_(kv[1].contiguous(), non_blocking=True)
            self.cpu_data_id = cpu_data_id
        else:
            if kv[0].is_cuda:
                cpu_data = tuple(_t.contiguous().to("cpu", non_blocking=True) for _t in kv)
            else:
                cpu_data = tuple(_t.contiguous() for _t in kv)

            if pin_memory:
                cpu_data = tuple(_t.pin_memory() for _t in cpu_data)
        
        if load_to_cache:
            gpu_data, gpu_data_id = cache.alloc()
            gpu_data = gpu_data.view((2, num_heads_kv, self.cache.max_block_size, dim_head))
            gpu_data[0][:, :size, :].copy_(kv[0], non_blocking=True)
            gpu_data[1][:, :size, :].copy_(kv[1], non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
        else:
            gpu_data, gpu_data_id = None, None
            event = None

        self.cpu_data = cpu_data
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id
        self.event = event
        self.size = size
        self.num_heads_kv = num_heads_kv
        self.dim_head = dim_head
        self.pin_memory = pin_memory

        if allow_disk_offload is not False:
            # set up disk offloading
            self.allow_disk_offload = True
            self.id = MemoryBlock._instance_counter
            MemoryBlock._instance_counter += 1
            self.offload_dir = offload_dir
            os.makedirs(self.offload_dir, exist_ok=True)
            if load_to_disk:
                self.offload_to_disk()
        else:
            self.allow_disk_offload = False
        
    def __del__(self):
        if hasattr(self, 'allow_disk_offload') and self.allow_disk_offload:
            self._delete_from_disk()

    def load(self, target: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, load_cache: bool = True) -> bool:
        if self.cpu_data is None:
            assert self.on_disk, "CPU data is None but on_disk is also set to False"
            self._load_from_disk()
        num_heads_kv, _, dim_head = self.cpu_data[0].shape
        assert target[0].shape == (num_heads_kv, self.size, dim_head)

        if self.gpu_data is not None:
            if target is not None:
                target[0].copy_(self.gpu_data[0][:, :self.size, :], non_blocking=True)
                target[1].copy_(self.gpu_data[1][:, :self.size, :], non_blocking=True)
                target_event = torch.cuda.Event()
                target_event.record(torch.cuda.current_stream())
            else:
                target_event = None

            return False, target_event

        gpu_data, gpu_data_id = self.cache.alloc()
        gpu_data = gpu_data.view((2, num_heads_kv, self.cache.max_block_size, dim_head))

        if target is not None:
            target[0].copy_(self.cpu_data[0][:, :self.size, :], non_blocking=True)
            target[1].copy_(self.cpu_data[1][:, :self.size, :], non_blocking=True)
            target_event = torch.cuda.Event()
            target_event.record(torch.cuda.current_stream())
            gpu_data[0][:, :self.size, :].copy_(target[0], non_blocking=True)
            gpu_data[1][:, :self.size, :].copy_(target[1], non_blocking=True)
        else:
            gpu_data[0][:, :self.size, :].copy_(self.cpu_data[0][:, :self.size, :], non_blocking=True)
            gpu_data[1][:, :self.size, :].copy_(self.cpu_data[1][:, :self.size, :], non_blocking=True)

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        self.event = event
        self.gpu_data = gpu_data
        self.gpu_data_id = gpu_data_id

        return True, target_event    

    def get(self):
        assert self.gpu_data is not None
        self.event.wait()
        return self.gpu_data[:, :, :self.size, :]

    def offload(self):
        assert self.gpu_data is not None
        self.event.wait()
        self.gpu_data = None
        self.cache.delete(self.gpu_data_id)
        self.gpu_data_id = None

    def offload_to_disk(self):
        if not self.on_disk:
            torch.save(self.cpu_data[0][:, :self.size, :].clone(), os.path.join(self.offload_dir, f"0/{self.id}.pt"), pickle_protocol=4)
            torch.save(self.cpu_data[1][:, :self.size, :].clone(), os.path.join(self.offload_dir, f"1/{self.id}.pt"), pickle_protocol=4)
            self.on_disk = True
        self.cpu_data = None
        self.cpu_cache.delete(self.cpu_data_id)
        self.cpu_data_id = None

    def _load_from_disk(self):
        self.cpu_data, self.cpu_data_id = self.cpu_cache.alloc()
        self.cpu_data = self.cpu_data.view((2, self.num_heads_kv, self.cache.max_block_size, self.dim_head))
        self.cpu_data[0, :, :self.size, :].copy_(torch.load(os.path.join(self.offload_dir, f"0/{self.id}.pt"), map_location='cpu'), non_blocking=True)
        self.cpu_data[1, :, :self.size, :].copy_(torch.load(os.path.join(self.offload_dir, f"1/{self.id}.pt"), map_location='cpu'), non_blocking=True)
    
    def _delete_from_disk(self):
        if self.on_disk:
            os.remove(os.path.join(self.offload_dir, f"0/{self.id}.pt"))
            os.remove(os.path.join(self.offload_dir, f"1/{self.id}.pt"))
        

class VectorTensor:
    def __init__(
        self, 
        hidden_size,
        element_dtype,
        layer_idx,
        device='cuda'
    ):
        init_cached_size = 16
        self.data = torch.empty(
            (init_cached_size, hidden_size),
            dtype=element_dtype,
            device=device
        )
        self.length = 0
        self.cache_size = init_cached_size
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

    def append_cache(self):
        new_cache_size = self.cache_size + 128
        data_shape = self.data.shape
        new_data = torch.empty(
            (new_cache_size,) + data_shape[1:],
            device=self.data.device,
            dtype=self.data.dtype
        )
        new_data[:self.cache_size,...].copy_(self.data)
        self.data = new_data
        self.cache_size = new_cache_size

    def append(self, tensor: torch.Tensor):
        assert tensor.dtype == self.data.dtype
        assert tensor.size(1) == self.hidden_size
        assert tensor.is_contiguous()

        append_l = tensor.size(0)

        while self.length + append_l > self.cache_size:
            self.append_cache()

        self.data[self.length: self.length+append_l, ...].copy_(tensor)

        self.length += append_l

    def get_data(self):
        return self.data[:self.length, ...]

    def get_similarity(self, tensor: torch.Tensor):
        assert tensor.dim() == 1 and tensor.size(0) == self.hidden_size
        logits = torch.matmul(self.data[:self.length], tensor[:, None].to(self.data.device)).squeeze(dim=-1)
        assert logits.dim() == 1 and logits.size(0) == self.length
        return logits

    def get_topk(self, tensor: torch.Tensor, topk):
        logits = self.get_similarity(tensor)
        return logits.topk(topk, dim=0).indices

    def sort_by_similarity(self, tensor: torch.Tensor):
        logits = self.get_similarity(tensor)
        return torch.sort(logits, descending=True).indices.cpu().tolist()

    def __len__(self):
        return self.length


GLOBAL_STREAM = None

class ContextManager:

    def __init__(self, 
                 layer_idx,
                 position_embedding,
                 n_init, 
                 n_local, 
                 max_block_size, 
                 max_cached_block, 
                 exc_block_size, 
                 min_block_size: int = 1,
                 async_global_stream: bool = True,
                 pin_memory: bool = False,
                 perhead: bool = False,
                 repr_topk: int = 1,
                 surprisal_threshold_gamma: float = 1.1,
                 n_mem = 2048,
                 uniform_blocks: bool = False,
                 random_topk_blocks: bool = False,
                 similarity_refinement: bool = False,
                 refine_with_buffer: bool = False,
                 refine_from_layer: int = 0,
                 similarity_metric: str = 'modularity',
                 use_contiguity_buffer: bool = False,
                 contiguity_buffer_size: float = 0.3,
                 use_hf_acc: bool = False,
                 disk_offload_dir: str = "./offload_data",
                 allow_disk_offload: bool = False,
                 vector_offload: bool = False,
                 **kwargs
    ):
        
        self.length = 0
        self.position_embedding = position_embedding
        self.n_init = n_init                                                # attention sink initial tokens
        self.n_local = n_local                                              # local sliding window size
        self.max_block_size = max_block_size                                # max memory block size
        self.min_block_size = min_block_size
        self.repr_topk = repr_topk                                          # number of representative tokens per memory block, L_k
        self.max_cached_block = max_cached_block                            # maximum number of memory blocks stored in GPU memory.  
        self.exc_block_size = exc_block_size                                # chunk size
        assert exc_block_size <= n_local                                    # no global token in input
        self.Attn = TorchMultiStageDotProductAttention
        self.initialized = False
        self.load_count = 0                                                 # total memory blocks retrieved across all timesteps
        self.async_global_stream = async_global_stream
        self.pin_memory = pin_memory
        self.perhead = perhead                                              # create separate mem unit per head
        self.global_context_cap = n_init + exc_block_size + n_mem
        self.surprisal_threshold_gamma = surprisal_threshold_gamma
        self.layer_idx = layer_idx
        self.random_topk_blocks = random_topk_blocks
        self.uniform_blocks = uniform_blocks
        self.similarity_refinement = similarity_refinement
        self.refine_with_buffer = refine_with_buffer
        self.refine_from_layer = refine_from_layer
        self.similarity_metric = similarity_metric
        self.use_contiguity_buffer = use_contiguity_buffer
        self.contiguity_buffer_size = contiguity_buffer_size

        self.use_hf_acc = use_hf_acc
        self.disk_offload_dir = disk_offload_dir
        self.allow_disk_offload = False if allow_disk_offload is False else None
        self.vector_offload = vector_offload
        if self.allow_disk_offload is None:
            self.min_free_cpu_memory = kwargs.get("min_free_cpu_memory", 100)  # minimum memory in GB to keep free on CPU during memory block allocation
            world_size = 4 if torch.cuda.device_count() == 1 else 2
            self.world_size = kwargs.get("world_size", world_size)

        global GLOBAL_STREAM
        if self.async_global_stream and GLOBAL_STREAM is None:
            GLOBAL_STREAM = torch.cuda.Stream()

        self.max_total_retrieved_tokens = self.global_context_cap - exc_block_size - n_init
        assert self.max_total_retrieved_tokens <= max_cached_block*min_block_size, f"Not enough cached blocks to fit {self.max_total_retrieved_tokens} tokens."

    def _init(
        self, 
        local_q, local_k, local_v,
        global_q, global_k, global_v
    ):
        assert local_q.dim() == 4
        batch_size, num_heads, len_q, dim_head = local_q.shape
        num_heads_kv = local_k.size(1)

        for _t in [local_q, local_k, local_v, global_q, global_k, global_v]:
            assert _t.size(0) == batch_size
            assert (_t.size(1) == num_heads or _t.size(1) == num_heads_kv)
            assert _t.size(2) == len_q
            assert _t.size(3) == dim_head
            assert _t.is_cuda

        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv
        self.unit_size_kv = num_heads_kv
        self.dim_head = dim_head

        self.global_blocks = [[] for _ in range(self.batch_size)] # [[memory_block]]
        self.cached_blocks = [{} for _ in range(self.batch_size)] # [{block_id: timestep_last_use}]
        if self.allow_disk_offload is not False:
            self.block_usage = [{} for _ in range(self.batch_size)] # [{block_id: timestep_last_use}]
        self.num_global_block = 0 # number of memory blocks

        if self.use_contiguity_buffer:
            self.contiguity_buffer = [[] for _ in range(self.batch_size)]

        # mean of repr tokens per block, 1/L_k Σ k^B
        self.block_repr_k = [VectorTensor(
            dim_head * self.num_heads, global_k.dtype, self.layer_idx, device=local_k.device
        ) for _ in range(self.batch_size)]

        # local kv cache
        self.local_k = torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=local_k.dtype, device=local_k.device)
        self.local_v = torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=local_v.dtype, device=local_v.device)

        self.global_remainder = (
            torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device),
            torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=global_v.dtype, device=global_v.device),
            torch.empty((self.batch_size, self.num_heads, 0, dim_head), dtype=global_q.dtype, device=global_q.device),
        )

        self.global_remainder_surprisal = torch.empty((self.batch_size, 0), dtype=global_k.dtype, device=global_k.device)
        self.global_remainder_repr_score = torch.empty((self.batch_size, self.num_heads, 0), dtype=global_k.dtype, device=global_k.device)
        self.global_block_divide = torch.empty((self.batch_size, 0), dtype=torch.bool, device=global_k.device)
        self.global_remainder_repr_score_buffer = torch.empty((self.batch_size, self.num_heads, self.max_block_size), dtype=global_k.dtype, device=global_k.device)
        self.global_remainder_k_buffer = torch.empty((self.batch_size, self.num_heads_kv, self.max_block_size, self.dim_head), dtype=global_k.dtype, device=global_k.device)

        # initial kv tokens
        self.init_k = torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)
        self.init_v = torch.empty((self.batch_size, self.num_heads_kv, 0, dim_head), dtype=global_k.dtype, device=global_k.device)        
        self.init_exc = False

        self.dtype = local_q.dtype
        self.position_embedding._update_cos_sin_tables_len(
            self.n_local + self.exc_block_size + 1, local_k.device, local_k.dim()
        )

        # retrieved kv memory alloc
        buffer_len = self.global_context_cap + 2 * self.max_block_size
        self.global_buffer = torch.empty(
                (2, self.batch_size, self.num_heads_kv, buffer_len, dim_head),
                dtype = global_k.dtype, device=global_k.device
            )
        self.global_buffer_init_st = 0
        self.global_buffer_init_ed = 0

        # cached memory blocks
        cuda_cache_device = local_k.device if self.use_hf_acc else 'cuda'
        self.cuda_cache = CudaCache(
            self.max_cached_block * self.batch_size,
            self.unit_size_kv * self.max_block_size * dim_head * 2,
            self.max_block_size,
            local_k.dtype,
            device=cuda_cache_device
        )
        self.create_memory_block = functools.partial(MemoryBlock, pin_memory = self.pin_memory, offload_dir = self.disk_offload_dir)
        
        self.cpu_cache = None
        
        self.global_length = [0 for b in range(self.batch_size)]
        self.initialized = True

    def _init_cpu_cache(self, local_q, local_k):
        _, _, _, dim_head = local_q.shape
        max_cpu_cache_memory = int(max(self.min_free_cpu_memory*(1024**3), (psutil.virtual_memory().available - self.min_free_cpu_memory*(1024**3))/self.world_size))
        token_size = local_k[:, 0, :].element_size() * local_k[:, 0, :].numel()
        self.max_cpu_cached_blocks = max_cpu_cache_memory // (token_size * 2 * self.max_block_size)
        if self.layer_idx == 0:
            print(f"Initialising CPU cache. Number of blocks allocated: {self.max_cpu_cached_blocks}")
        self.cpu_cache = CudaCache(
            self.max_cpu_cached_blocks * self.batch_size,
            self.unit_size_kv * self.max_block_size * dim_head * 2,
            self.max_block_size,
            local_k.dtype,
            device=torch.device('cpu')
        )
    
    def _offload_vector(self):
        if self.layer_idx == 0:
            print("Offloading VectorTensor to CPU to run single-GPUs on long-contexts.")
        for u in range(self.batch_size):
            self.block_repr_k[u].data = self.block_repr_k[u].data.to(torch.device('cpu'))

    def _num_memory_blocks(self):
        return len(self.global_blocks[0])
    
    def _remove_lru_blocks(self, u, num_remove: Optional[int] = None, ignore_blocks = None):
        # removes least recently used blocks
        if num_remove is None:
            tokens_in_cache = sum([self.global_blocks[u][bidx].size for bidx in self.cached_blocks[u].keys()])
            num_remove = tokens_in_cache - self.max_total_retrieved_tokens

        if num_remove <= 0:
            return

        lst = list(self.cached_blocks[u].items())
        lst.sort(key=lambda x: x[1])

        removed = 0
        for i in range(len(lst)):
            idx = lst[i][0]
            if ignore_blocks is None or (idx not in ignore_blocks):
                self.global_blocks[u][idx].offload()
                self.cached_blocks[u].pop(idx)
                removed += self.global_blocks[u][idx].size

            if removed >= num_remove:
                return
            
    def _remove_cpu_lru_blocks(self, ignore_blocks = None):
        num_remove = ((self.exc_block_size + self.max_total_retrieved_tokens) // self.min_block_size) + 1 - len(self.cpu_cache.idle_set)
        if num_remove > 0:
            for u in range(self.batch_size):
                lst = list(self.block_usage[u].items())
                lst.sort(key=lambda x: x[1])
                removed = 0
                for i in range(len(lst)):
                    idx = lst[i][0]
                    if (ignore_blocks is None or idx not in ignore_blocks) and self.global_blocks[u][idx].cpu_data is not None:
                        self.global_blocks[u][idx].offload_to_disk()
                        removed += 1

                        if removed >= num_remove:
                            break
    
    def _calc_topk_blocks(self, len_q, global_q):
        global_remainder_len = max(self._global_remainder_ed - self._global_remainder_st + len_q - self.n_local, 0)
        global_context_cap = self.global_context_cap - global_remainder_len - (self.length > self.n_local) * self.init_k.size(-2)
        if self.use_contiguity_buffer:
            if self.contiguity_buffer_size < 1:
                global_context_cap -= int(self.contiguity_buffer_size*global_context_cap + 1)
            else:
                global_context_cap -= self.contiguity_buffer_size
        
        if self.global_length[0] <= global_context_cap:
            return [list(range(len(self.global_blocks[0]))) for _ in range(self.batch_size)]
        
        global_q = global_q.mean(dim=2, keepdim=False)
        assert global_q.shape == (self.batch_size, self.num_heads, self.dim_head)
        global_q = global_q.reshape(self.batch_size, self.dim_head * self.num_heads)

        retrieved_blocks = []
        for u in range(self.batch_size):
            
            if self.random_topk_blocks:
                sorted_block_idx = list(range(self.num_global_block))
                random.shuffle(sorted_block_idx)
            else:
                sorted_block_idx = self.block_repr_k[u].sort_by_similarity(global_q[u])
            sorted_block_idx = iter(sorted_block_idx)
            context_len = 0

            filled_global_context = False
            batch_retrieved_blocks = []
            while not filled_global_context:
                b_idx = next(sorted_block_idx, None)
                cur_blocks = [b_idx]

                if b_idx is None:
                    filled_global_context = True
                    break
                
                for cur in cur_blocks:
                    if cur in batch_retrieved_blocks or cur < 0 or cur > self.num_global_block - 1:
                        continue
                    batch_retrieved_blocks.append(cur)
                    prev_context_len = context_len
                    context_len += self.global_blocks[u][cur].size

                    if context_len >= global_context_cap:
                        if abs(global_context_cap - prev_context_len) <= abs(context_len - global_context_cap):
                            batch_retrieved_blocks.pop()
                            context_len -= self.global_blocks[u][cur].size
                        filled_global_context = True
                        break

            retrieved_blocks.append(batch_retrieved_blocks)

        return retrieved_blocks
    
    def _update_contiguity_buffer(self, len_q, topk_blocks):
        global_remainder_len = max(self._global_remainder_ed - self._global_remainder_st + len_q - self.n_local, 0)
        topk_global_context_cap = self.global_context_cap - global_remainder_len - (self.length > self.n_local) * self.init_k.size(-2)
        if self.contiguity_buffer_size  < 1:
            ctg_global_context_cap = int(self.contiguity_buffer_size*topk_global_context_cap + 1)
        else:
            ctg_global_context_cap = self.contiguity_buffer_size
        
        # get all directly adjacent blocks to topk blocks, within allowed context cap
        for u in range(self.batch_size):
            if len(topk_blocks[u]) == 0:
                continue
            batch_ctg_blocks = []
            context_len = 0
            filled_global_context = False
            batch_topk_blocks = iter(topk_blocks[u])
            while not filled_global_context:
                bidx = next(batch_topk_blocks, None)
                if bidx is None:
                    break
                cur_blocks = [bidx+1, bidx-1]

                for cur in cur_blocks:
                    if cur in batch_ctg_blocks or cur in topk_blocks[u] or cur < 0 or cur > self.num_global_block - 1:
                        continue
                    batch_ctg_blocks.append(cur)
                    prev_context_len = context_len
                    context_len += self.global_blocks[u][cur].size

                    if context_len >= ctg_global_context_cap:
                        if abs(ctg_global_context_cap - prev_context_len) <= abs(context_len - ctg_global_context_cap):
                            batch_ctg_blocks.pop()
                            context_len -= self.global_blocks[u][cur].size
                        filled_global_context = True
                        break
            
            # update contiguity buffer
            batch_ctg_blocks.reverse()
            if len(batch_ctg_blocks) > 0:
                if len(self.contiguity_buffer[u]) == 0 or filled_global_context:
                    self.contiguity_buffer[u] = batch_ctg_blocks
                else:
                    total_buffer_len = sum([self.global_blocks[u][b].size for b in self.contiguity_buffer[u]])
                    if total_buffer_len + context_len < ctg_global_context_cap:
                        self.contiguity_buffer[u] += batch_ctg_blocks
                    else:
                        past_blocks = self.contiguity_buffer[u]
                        past_blocks.reverse()
                        self.contiguity_buffer[u] = batch_ctg_blocks
                        for cur in past_blocks:
                            if cur in batch_ctg_blocks or cur in topk_blocks[u] or cur < 0 or cur > self.num_global_block - 1:
                                continue
                            self.contiguity_buffer[u].insert(0,cur)
                            prev_context_len = context_len
                            context_len += self.global_blocks[u][cur].size

                            if context_len >= ctg_global_context_cap:
                                if abs(ctg_global_context_cap - prev_context_len) <= abs(context_len - ctg_global_context_cap):
                                    self.contiguity_buffer[u].pop(0)
                                    context_len -= self.global_blocks[u][cur].size
                                break

    def _get_init_and_remainder_context(self, init_st, global_h_k, global_h_v, global_remainder_len):
        init_len = self.init_k.size(-2)
        init_ed = init_st + init_len
        if self.length > self.n_local:
            global_h_k[:, :, init_st: init_ed, :].copy_(self.init_k, non_blocking=True)
            global_h_v[:, :, init_st: init_ed, :].copy_(self.init_v, non_blocking=True)
        
        ed = init_ed

        rmd_st = init_ed
        rmd_ed = rmd_st + global_remainder_len
        ed = rmd_ed

        global_h_k[:, :, rmd_st: rmd_ed, :].copy_(self.global_remainder[0][:, :, self._global_remainder_st:self._global_remainder_st+global_remainder_len, :], non_blocking=True)
        global_h_v[:, :, rmd_st: rmd_ed, :].copy_(self.global_remainder[1][:, :, self._global_remainder_st:self._global_remainder_st+global_remainder_len, :], non_blocking=True)

        sliding_window = (self.global_remainder[0].size(-2) + rmd_st, self.n_local)

        global_h_k = global_h_k[:, :, :ed, :]
        global_h_v = global_h_v[:, :, :ed, :]

        return global_h_k, global_h_v, sliding_window

    def _get_global_hidden_and_mask(
        self, len_q, topk_blocks
    ):
        # TODO: must refactor to account for batch sizes larger than 1
        assert len(topk_blocks) == self.batch_size
        global_remainder_len = max(self._global_remainder_ed - self._global_remainder_st + len_q - self.n_local, 0)

        sliding_window = None

        global_h_k = self.global_buffer[0]
        global_h_v = self.global_buffer[1]

        num_retrieved_blocks = len(topk_blocks[0])

        size = 0
        for u in range(self.batch_size):
            assert len(topk_blocks[u]) == num_retrieved_blocks
            topk_blocks[u].sort()
            st = 0
            ed = 0
            for b_idx in topk_blocks[u]:
                assert b_idx in self.cached_blocks[u]
                ed = st + self.global_blocks[u][b_idx].size
                self.global_blocks[u][b_idx].load((global_h_k[u, :, st:ed, :], global_h_v[u, :, st:ed, :]))
                size += self.global_blocks[u][b_idx].size
                st = ed
        
        init_st = st
        global_h_k, global_h_v, sliding_window = self._get_init_and_remainder_context(init_st, global_h_k, global_h_v, global_remainder_len)

        return global_h_k, global_h_v, sliding_window, init_st

    def _retrieve_and_attend(
        self, local_q, local_k, local_v, global_q
    ):
        """retrieve top k memory blocks and compute attention outputs""" 
        local_h_q, local_h_k = self.position_embedding(local_q, local_k)
        local_h_v = local_v
        if self.use_hf_acc:
            local_h_q = local_h_q.to(local_q.device)
            local_h_k = local_h_k.to(local_k.device)
        # calc local result first to overlap host-device communication
        attn = self.Attn(local_h_q.shape, local_h_q.dtype, local_h_q.device)
        attn.append(
            local_h_q, local_h_k, local_h_v, 
            get_score=True, sliding_window=self.n_local
        )

        # get topk blocks and load to GPU cache
        with torch.cuda.stream(GLOBAL_STREAM):
            topk_blocks = self._calc_topk_blocks(local_h_q.size(-2), global_q)

            if self.use_contiguity_buffer:
                self._update_contiguity_buffer(local_h_q.size(-2), topk_blocks)
                for u in range(self.batch_size):
                    if len(self.contiguity_buffer[u]) > 0:
                        topk_blocks[u] = self.contiguity_buffer[u] + topk_blocks[u]

            self.load_count += 1
            for u in range(self.batch_size):
                tokens_in_cache = sum([self.global_blocks[u][bidx].size for bidx in self.cached_blocks[u].keys()])
                num_remove = tokens_in_cache - self.max_total_retrieved_tokens
                for b_idx in topk_blocks[u]:
                    if b_idx not in self.cached_blocks[u]:
                        num_remove += self.global_blocks[u][b_idx].size

                # update memory block cache
                self._remove_lru_blocks(u, num_remove, topk_blocks[u])
                if self.allow_disk_offload is True:
                    self._remove_cpu_lru_blocks(set(list(self.cached_blocks[u].keys())))

                for bidx in topk_blocks[u]:
                    self.cached_blocks[u][bidx] = self.load_count
                    if self.allow_disk_offload is not False:
                        self.block_usage[u][bidx] = self.load_count

            global_h_q = global_q
            global_h_k, global_h_v, global_sliding_window, init_st = self._get_global_hidden_and_mask(local_h_q.size(-2), topk_blocks)

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        # calc global result
        attn.append(
            global_h_q, global_h_k, global_h_v, 
            end=True, get_score=False, 
            sliding_window=global_sliding_window,
            complement_sliding_window=True
        )

        attn_output, repr_score = attn.get_result()

        self.exc_repr_score = repr_score[0]

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        self.attn = None
            
        return attn_output.view((self.batch_size, self.num_heads, -1, self.dim_head))

    def _from_group_kv(self, tensor):
        # reshapes kv if num_heads != num_heads_kv
        assert tensor.dim() == 3
        if tensor.size(0) == self.num_heads:
            return tensor
        _, length, dim_head = tensor.shape
        num_group = self.num_heads // self.num_heads_kv
        tensor = tensor.view((self.num_heads_kv, 1, length, dim_head))
        tensor = tensor.expand((self.num_heads_kv, num_group, length, dim_head)).reshape((self.num_heads, length, dim_head))
        return tensor
    
    def get_block_k(self, k, repr_score):
        # get topk representative tokens in memory block
        assert isinstance(repr_score, torch.Tensor)
        assert k.dim() >= 2
        k = self._from_group_kv(k)
        assert k.shape[:-1] == repr_score.shape
        repr_topk = min(self.repr_topk, repr_score.shape[-1])
        score_topk = repr_score.topk(repr_topk, dim=-1).indices
        assert score_topk.shape == (self.num_heads, repr_topk)
        return torch.gather(k, -2, score_topk[:, :, None].expand(self.num_heads, repr_topk, self.dim_head)), repr_topk
        
    def _add_block(self, u, remainder_st, remainder_ed, load_to_disk=False):

        kv = (
            self.global_remainder[0][u, :, remainder_st: remainder_ed, :],
            self.global_remainder[1][u, :, remainder_st: remainder_ed, :]
        )

        self.global_blocks[u].append((
            self.create_memory_block(
                kv=kv,
                cache=self.cuda_cache,
                load_to_cache=False,
                allow_disk_offload=self.allow_disk_offload,
                load_to_disk=load_to_disk,
                cpu_cache=self.cpu_cache
            )
        ))
        if self.allow_disk_offload is not False:
            bidx = len(self.global_blocks[u]) - 1
            self.block_usage[u][bidx] = 0
        
        global_block_repr_k, repr_topk = self.get_block_k(
            self.global_remainder[0][u, :, remainder_st: remainder_ed, :],
            self.global_remainder_repr_score[u, :, remainder_st: remainder_ed]
        )

        assert global_block_repr_k.shape == (self.num_heads, repr_topk, self.dim_head)
        global_block_repr_k = global_block_repr_k.mean(dim=-2, keepdim=False)
        global_block_repr_k = global_block_repr_k.reshape(self.num_heads * self.dim_head)
        global_block_repr_k = global_block_repr_k[None, :]
        self.block_repr_k[u].append(global_block_repr_k)

        self.num_global_block += 1
        self.global_length[u] += remainder_ed - remainder_st

    def append(
        self,
        local_q, local_k, local_v,
        global_q, global_k, global_v,
    ):
        batch_size = local_q.size(0)
        input_length = local_q.size(-2)
        assert input_length <= self.exc_block_size
        assert batch_size == 1

        if self.perhead:
            num_heads = local_q.size(1)
            num_heads_kv = local_v.size(1)
            def repeat_kv(t):
                t = t.view(batch_size, num_heads_kv, 1, input_length, -1)
                t = t.expand(batch_size, num_heads_kv, num_heads // num_heads_kv, input_length,  -1)
                t = t.reshape(batch_size * num_heads, 1, input_length, -1)
                return t

            local_q = local_q.view(batch_size * num_heads, 1, input_length, -1)
            local_k = repeat_kv(local_k)
            local_v = repeat_kv(local_v)
            global_q = global_q.view(batch_size * num_heads , 1, input_length, -1)
            global_k = repeat_kv(global_k)
            global_v = repeat_kv(global_v)

        if not self.initialized:
            self._init(
                local_q, local_k, local_v,
                global_q, global_k, global_v
            )

        if self.allow_disk_offload is True:
            if self.cpu_cache is None:
                self._init_cpu_cache(local_q, local_k)
            if self.vector_offload and self.block_repr_k[0].data.device != torch.device('cpu'):
                self._offload_vector()

        input_length = local_q.size(-2)

        if self.async_global_stream:
            GLOBAL_STREAM.wait_stream(torch.cuda.current_stream())

        # concat local kv cache to inputs
        self.local_k = torch.cat((self.local_k, local_k), dim=-2)
        self.local_v = torch.cat((self.local_v, local_v), dim=-2)
        self.kv_length = self.local_k.size(-2)

        # append global remainder
        with torch.cuda.stream(GLOBAL_STREAM):
            global_q = self.position_embedding.apply_rotary_pos_emb_one_angle(
                global_q, self.n_local
            )

            self._global_remainder_st = 0
            self._global_remainder_ed = self.global_remainder[0].size(-2)

            self.global_remainder = (
                torch.cat((self.global_remainder[0], global_k), dim=-2),
                torch.cat((self.global_remainder[1], global_v), dim=-2),
                torch.cat((self.global_remainder[2], global_q), dim=-2)
            )

            self.global_remainder_repr_score = torch.cat(
                (self.global_remainder_repr_score, 
                torch.zeros(
                        (self.batch_size, self.num_heads, global_k.size(-2)),
                        dtype=global_k.dtype, device=global_k.device
                    )  
                ),
                dim=-1
            )

            self.global_remainder_surprisal = torch.cat(
                (self.global_remainder_surprisal, 
                torch.zeros(
                        (self.batch_size, global_k.size(-2)),
                        dtype=global_k.dtype, device=global_k.device
                    )  
                ),
                dim=-1
            )

            self.global_block_divide = torch.cat(
                (self.global_block_divide, 
                torch.zeros(
                        (self.batch_size, global_k.size(-2)),
                        dtype=global_k.dtype, device=global_k.device
                    )  
                ),
                dim=-1
            )

        # retrieve top k memory blocks and compute attention outputs 
        attn_output = self._retrieve_and_attend(
            local_q,
            self.local_k,
            self.local_v,
            global_q,
        )

        if self.perhead:
            attn_output = attn_output.view(batch_size, num_heads, input_length, -1)

        return attn_output

        
    def update_memory(
        self, exc_length, exc_surprisal, surprisal_values=None
    ):
        with torch.cuda.stream(GLOBAL_STREAM):

            global_remainder_ed = self._global_remainder_ed + exc_length
            global_remainder_st = self._global_remainder_st

            global_remainder_len = global_remainder_ed - global_remainder_st

            assert self.exc_repr_score.shape[:3] == (self.batch_size, self.num_heads, self.kv_length)
            self.exc_repr_score = self.exc_repr_score[:, :, -exc_length-self.n_local:]
            self.global_remainder_repr_score[:, :, global_remainder_ed-self.exc_repr_score.size(-1):global_remainder_ed].add_(self.exc_repr_score)

            if exc_surprisal is not None:
                if self.use_hf_acc:
                    exc_surprisal = exc_surprisal.to(self.global_remainder_surprisal.device)
                if not self.uniform_blocks:
                    assert exc_surprisal.shape == (self.batch_size, exc_length)
                    if surprisal_values is None:   
                        self.global_remainder_surprisal[:, global_remainder_ed-exc_length:global_remainder_ed].copy_(exc_surprisal)
                    else:
                        if self.use_hf_acc:
                            surprisal_values = surprisal_values.to(self.global_remainder_surprisal.device)
                        self.global_remainder_surprisal[:, global_remainder_ed-exc_length:global_remainder_ed].copy_(surprisal_values)

                    if exc_surprisal.dtype == torch.bool:
                        divide = exc_surprisal         
                    else:
                        # refactor - more efficient to bring outside the contextmanager as the same computation is happening per layer
                        avg_st = max(global_remainder_ed - self.n_local, 0)
                        avg_ed = max(global_remainder_ed - exc_length, self.n_init)
                        divide = exc_surprisal > self.surprisal_threshold_gamma * torch.std(self.global_remainder_surprisal[:, avg_st:avg_ed], dim=-1) + torch.mean(self.global_remainder_surprisal[:, avg_st:avg_ed], dim=-1)
                else:
                    divide = torch.zeros(exc_surprisal.shape, dtype=torch.bool)
                    divide[:, ::self.max_block_size] = True

                self.global_block_divide[:, global_remainder_ed-exc_length:global_remainder_ed].copy_(divide)

            if not self.init_exc and global_remainder_len > self.n_local:
                global_k = self.global_remainder[0]
                global_v = self.global_remainder[1]

                append_init_len = min(
                    self.n_init - self.init_k.size(-2),
                    global_remainder_len - self.n_local
                )
                self.init_k = torch.cat(
                    (self.init_k, global_k[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
                )
                self.init_v = torch.cat(
                    (self.init_v, global_v[:, :, global_remainder_st:global_remainder_st + append_init_len, :]), dim=-2
                )
                global_remainder_st += append_init_len
                global_remainder_len -= append_init_len

                if self.init_k.size(-2) == self.n_init:
                    self.init_exc = True

            if global_remainder_len >= self.n_local + self.min_block_size:
     
                ed = global_remainder_len - self.n_local
                for u in range(self.batch_size):
    
                    divide = self.global_block_divide[u, global_remainder_st : global_remainder_st + ed]
                    surprising_token_idx = torch.where(divide > 0)[0]
                    
                    if surprising_token_idx.shape[-1] == 0:
                        if divide.shape[-1] > self.max_block_size:
                            surprising_token_idx = torch.tensor(
                                range(0, divide.shape[-1], self.max_block_size), 
                                dtype=torch.int16, 
                                device=divide.device
                            ) 
                        else: 
                            surprising_token_idx = torch.tensor(
                                [divide.shape[-1]], 
                                dtype=torch.int16, 
                                device=divide.device
                            ) 

                    surprising_token_idx = torch.cat((torch.tensor([0], device = surprising_token_idx.device), surprising_token_idx, torch.tensor([len(divide)], device = surprising_token_idx.device)))
                    block_sizes = surprising_token_idx[1:] - surprising_token_idx[:-1]
                    mask = torch.where(block_sizes != 0)[0]
                    block_sizes = block_sizes[mask]
                    load_to_disk = False
                    if self.allow_disk_offload is True and len(self.cpu_cache.idle_set) == 1:
                        print("=== WARNING! ===> ONLY ONE CPU CACHE UNIT LEFT! OFFLOADING DIRECTLY TO DISK")
                        load_to_disk = True
                    acc_b = 0
                    for i, b in enumerate(block_sizes):
                        acc_b += int(b.item())
                        while acc_b >= self.max_block_size:
                            self._add_block(u, global_remainder_st, global_remainder_st + self.max_block_size
                                            , load_to_disk=load_to_disk)
                            global_remainder_st += self.max_block_size
                            acc_b -= self.max_block_size
                        if acc_b < min(self.min_block_size, divide.shape[-1]):
                            continue
                        if acc_b > 0 and i != len(block_sizes)-1:
                            self._add_block(u, global_remainder_st, global_remainder_st + acc_b
                                            , load_to_disk=load_to_disk)
                            global_remainder_st += acc_b
                        acc_b = 0
            
            self._global_remainder_ed = global_remainder_ed
            self._global_remainder_st = global_remainder_st

        if self.async_global_stream:
            torch.cuda.current_stream().wait_stream(GLOBAL_STREAM)

        self.length += exc_length

        # update local and global tensor
        if self.local_k.size(-2) >= self.n_local:
            self.local_k = self.local_k[:, :, -self.n_local:, :]
            self.local_v = self.local_v[:, :, -self.n_local:, :]

        assert self._global_remainder_ed == self.global_remainder[0].size(-2)
        with torch.cuda.stream(GLOBAL_STREAM):
            self.global_remainder = (
                self.global_remainder[0][:, :, self._global_remainder_st:, :],
                self.global_remainder[1][:, :, self._global_remainder_st:, :],
                self.global_remainder[2][:, :, self._global_remainder_st:, :],
            )
            self.global_remainder_surprisal = self.global_remainder_surprisal[:, self._global_remainder_st:]
            self.global_block_divide = self.global_block_divide[:, self._global_remainder_st:]
            self.global_remainder_repr_score = self.global_remainder_repr_score[:, :, self._global_remainder_st:]

