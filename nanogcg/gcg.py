import os
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy
import gc
import logging

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import math
import torch
import transformers
from torch import Tensor
from transformers import set_seed

from nanogcg.utils import INIT_CHARS, find_executable_batch_size, get_nonascii_toks, mellowmax, check_type

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    
@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"
    prefix_update_frequency: int = 1
    weight: float = 1.
    num_location: int = 2
    max_new_tokens: int = 300
    inverse: bool = False
    sequential: bool = True
    gcg: bool = False

@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    best_generation: str
    best_success: bool
    all_generation: List[list]
    last_string: str
    last_generation: str
    last_success: bool
    vanilla_generation: str
    vanilla_success: bool
    losses: List[float]
    strings: List[str]
        
        
class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = [] # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.insert(0, (loss, optim_ids))
#             self.buffer.append((loss, optim_ids))
            return
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]
    
    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]
    
    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)

def sample_ids_from_grad(
    ids: Tensor, 
    grad: Tensor, 
    search_width: int, 
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized 
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization
    
    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device)
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids) 
            token ids 
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer
    
    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
           filtered_ids.append(ids[i]) 
    
    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
        )
    
    return torch.stack(filtered_ids)

class GCG:
    def __init__(
        self, 
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        self.batch_prefix_cache = None

        self.stop_flag = False

        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    
    def run(
        self,
        messages: Union[List[Union[str, List[dict]]], Union[str, List[dict]]],
        targets: Union[List[Union[str, List[str]]], str],
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        if isinstance(messages, str):
            messages = [[{"role": "user", "content": messages}]]
        elif check_type(messages, str):
            messages = [[{"role": "user", "content": message}] for message in messages]
        elif check_type(messages, dict):
            messages = [messages]
        messages = copy.deepcopy(messages)
        
        if isinstance(targets, str):
            self.targets = [[targets]] * len(messages)
        elif check_type(targets, str):
            self.targets = [[target] for target in targets]
        else:
            self.targets = [[target for target in target_list] for target_list in targets]

        self.batch_before_ids = [[]] * len(messages)
        self.batch_after_ids = [[]] * len(messages)
        self.batch_target_ids = [[]] * len(messages)
        self.batch_before_embeds = [[]] * len(messages)
        self.batch_after_embeds = [[]] * len(messages)
        self.batch_target_embeds = [[]] * len(messages)
        self.batch_prefix_cache = [[]] * len(messages)
        self.batch_all_generation = [[]] * len(messages)
        
        # Append the GCG string at the end of the prompt if location not specified
        for conversation in messages:
            if not any("{optim_str}" in message["content"] for message in conversation):
                conversation[-1]["content"] += "{optim_str}"
                
        self.messages = messages
        templates = [
            tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) if isinstance(message, list) else message
            for message in messages
        ]
        
        def bos_filter(template):
            if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
                return template.replace(tokenizer.bos_token, "")
            return template
        
        templates = list(map(bos_filter, templates))
        self.templates = templates
        self._update_target_and_context(config.optim_str_init)

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        
        embedding_layer = self.embedding_layer
        for _ in tqdm(range(config.num_steps)):
            
            if self.stop_flag:
                losses.append(buffer.get_lowest_loss())
                optim_ids = buffer.get_best_ids()
                optim_str = tokenizer.batch_decode(optim_ids)[0]
                optim_strings.append(optim_str)
                break
                
            # Compute the token gradient
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids) 
    
            with torch.no_grad():

                # Sample candidate token sequences based on the token gradient
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)

                new_search_width = sampled_ids.shape[0]
                
                # Compute loss on all candidate sequences 
                batch_size = new_search_width if config.batch_size is None else config.batch_size
                loss = 0
                for idx in range(len(self.batch_before_embeds)):
                    if self.batch_prefix_cache:
                        input_embeds = torch.cat([
                            embedding_layer(sampled_ids),
                            self.batch_after_embeds[idx].repeat(new_search_width, 1, 1),
                            *[
                                target_embedding.repeat(new_search_width, 1, 1) 
                                for target_embedding in self.batch_target_embeds[idx]
                            ]
#                             self.batch_target_embeds[idx].repeat(new_search_width, 1, 1),
                        ], dim=1)
                    else:
                        # If there's no prefix cache, or it's a general case not specific to batch
                        input_embeds = torch.cat([
                            self.batch_before_embeds[idx].repeat(new_search_width, 1, 1),
                            embedding_layer(sampled_ids),
                            self.batch_after_embeds[idx].repeat(new_search_width, 1, 1),
                            *[
                                target_embedding.repeat(new_search_width, 1, 1) 
                                for target_embedding in self.batch_target_embeds[idx]
                            ]
#                             self.batch_target_embeds[idx].repeat(new_search_width, 1, 1),
                        ], dim=1)
                    # Dynamic batch size execution for loss computation
                    loss += find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds, idx)

                # Average the total loss by the number of batches processed
                loss /= len(self.batch_before_embeds)  # Ensure this matches the batch index used above

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or len(buffer.buffer) < buffer.size or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)
                    
            print(f"Current Loss: {current_loss * 10:.4f}")
            
            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            
            buffer.log_buffer(tokenizer)                
            
            if (_ + 1) % self.config.prefix_update_frequency == 0:
                self._update_target_and_context(optim_str)
            else:
                for k, template in enumerate(self.templates):
                    # Generate the assistant's response
                    outputs, decoding_string = self.greedy_generation(template, optim_str, k)
                    print(decoding_string)
                    self.batch_all_generation[k].append((optim_str, decoding_string))
                    
            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.") 
                break
              
        min_loss_index = losses.index(min(losses)) 
        vanilla_generation= [self.greedy_generation(template, '')[1] for template in self.templates]
        best_generation= [self.greedy_generation(template, optim_strings[min_loss_index])[1]  for template in self.templates]
        last_generation = [self.greedy_generation(template, optim_strings[-1])[1]  for template in self.templates]
        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            best_generation = best_generation,
            best_success = [targets[i][0] not in best_generation[i] for i in range(len(best_generation))] if self.config.inverse else [targets[i][0] in best_generation[i] for i in range(len(best_generation))] ,
            last_string=optim_strings[-1],
            last_generation = last_generation,
            last_success = [targets[i][0] not in last_generation[i] for i in range(len(last_generation))] if self.config.inverse else [targets[i][0] in last_generation[i] for i in range(len(last_generation))] ,
            vanilla_generation = vanilla_generation,
            vanilla_success = [targets[i][0] not in best_generation[i] for i in range(len(vanilla_generation))] if self.config.inverse else [targets[i][0] in best_generation[i] for i in range(len(vanilla_generation))] ,
            all_generation = self.batch_all_generation,
            losses=losses,
            strings=optim_strings,
        )
        return result

    
    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
#             if config.buffer_size > 1:
#                 init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
#                 init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
#                 init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
#             else:
            init_buffer_ids = init_optim_ids
                
        else: # assume list
            if (len(config.optim_str_init) != config.buffer_size):
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = 1 # max(1, config.buffer_size) 

        # Compute the loss on the initial buffer entries
        init_buffer_losses = 0
        for idx in range(len(self.batch_before_embeds)):
            if self.batch_prefix_cache:
                init_buffer_embeds = torch.cat([
                    self.embedding_layer(init_buffer_ids),
                    self.batch_after_embeds[idx].repeat(true_buffer_size, 1, 1),
                    *[
                        target_embedding.repeat(true_buffer_size, 1, 1) 
                        for target_embedding in self.batch_target_embeds[idx]
                    ]
#                     self.batch_target_embeds[idx].repeat(true_buffer_size, 1, 1),
                ], dim=1)
            else:
                init_buffer_embeds = torch.cat([
                    self.batch_before_embeds[idx].repeat(true_buffer_size, 1, 1),
                    self.embedding_layer(init_buffer_ids),
                    self.batch_after_embeds[idx].repeat(true_buffer_size, 1, 1),
                    *[
                        target_embedding.repeat(true_buffer_size, 1, 1) 
                        for target_embedding in self.batch_target_embeds[idx]
                    ]
#                     self.batch_target_embeds[idx].repeat(true_buffer_size, 1, 1),
                ], dim=1)
            # Compute loss for each batch and collect
            init_buffer_losses += find_executable_batch_size(self.compute_candidates_loss, true_buffer_size)(init_buffer_embeds, idx)

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])
        
        buffer.log_buffer(tokenizer)
        logger.info("Initialized attack buffer.")
        
        return buffer
    
    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
        optim_ids : Tensor, shape = (1, n_optim_ids)
            the sequence of token ids that are being optimized 
        """
        model = self.model
        embedding_layer = self.embedding_layer

        # Create the one-hot encoding matrix of our optimized token ids
        optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
        optim_ids_onehot = optim_ids_onehot.to(dtype=model.dtype, device=model.device)
        optim_ids_onehot.requires_grad_()

        # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
        optim_embeds = optim_ids_onehot @ embedding_layer.weight

        # Calculate loss for each batch and average the gradients
        total_loss = 0
        for idx in range(len(self.batch_before_embeds)):
            if self.batch_prefix_cache:
                input_embeds = torch.cat([optim_embeds, self.batch_after_embeds[idx], *self.batch_target_embeds[idx]], dim=1)
                output = model(inputs_embeds=input_embeds, past_key_values=self.batch_prefix_cache[idx])
            else:
                input_embeds = torch.cat([self.batch_before_embeds[idx], optim_embeds, self.batch_after_embeds[idx], *self.batch_target_embeds[idx]], dim=1)
                output = model(inputs_embeds=input_embeds)
            
            logits = output.logits
            
#             shift = input_embeds.shape[1] - self.batch_target_ids[idx].shape[1]
#             shift_logits = logits[..., shift-1:-1, :].contiguous()  # (1, num_target_ids, vocab_size)
#             shift_labels = self.batch_target_ids[idx]
            
            shift = input_embeds.shape[1] - sum([target_embedding.shape[1] for target_embedding in self.batch_target_ids[idx]])
            current_shift = shift - 1
            for p in range(len(self.batch_target_embeds[idx])):
                length = self.batch_target_embeds[idx][p].shape[1]
                if p % 2 == 0:
                    position = list(range(current_shift, current_shift + length))
                    position = torch.tensor(position, dtype=torch.long).to(model.device)
                    shift_labels = self.batch_target_ids[idx][p]
                    shift_logits = logits[..., position, :].contiguous()  # (1, num_target_ids, vocab_size)

                    if self.config.use_mellowmax:
                        label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                        batch_loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    else:
                        batch_loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    total_loss += batch_loss * (self.config.weight ** (p//2))
                    
#                     if self.config.sequential and any(shift_logits[0].argmax(1) != shift_labels[0]):
#                         break
                        
                current_shift = current_shift + length
                
        # Average the loss over all batches
        mean_loss = total_loss / len(self.batch_before_embeds)

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[mean_loss], inputs=[optim_ids_onehot])[0]
        
        if self.config.inverse:
            optim_ids_onehot_grad = - optim_ids_onehot_grad

        return optim_ids_onehot_grad

    def compute_candidates_loss(
        self,
        search_batch_size: int, 
        input_embeds: Tensor, 
        idx: int
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
        """
        all_loss = []
        prefix_cache_batch = []
        
        if self.batch_prefix_cache:
            prefix_cache = self.batch_prefix_cache[idx]
        else:
            prefix_cache = None
            
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]
                
                if prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in prefix_cache[i]] for i in range(len(prefix_cache))]

                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - sum([target_embedding.shape[1] for target_embedding in self.batch_target_ids[idx]])
#                 shift_logits = logits[..., tmp-1:-1, :].contiguous()
#                 shift_labels = self.batch_target_ids[idx].repeat(current_batch_size, 1)
                
                positions = []
                all_shift_labels = []
                current_shift = tmp - 1
                for p in range(len(self.batch_target_embeds[idx])):
                    length = self.batch_target_embeds[idx][p].shape[1]
                    if p % 2 == 0:
                        positions.append(list(range(current_shift, current_shift + length)))
                        all_shift_labels.append(self.batch_target_ids[idx][p])
                    current_shift = current_shift + length
                
                current_loss = 0
                active_mask = torch.ones(current_batch_size, dtype=torch.bool, device=self.model.device)
                for p in range(len(positions)):
                    position = torch.tensor(positions[p], dtype=torch.long).to(self.model.device)
                    shift_logits = logits[..., position, :].contiguous() 
                    shift_labels = all_shift_labels[p].repeat(current_batch_size, 1)
                    
#                     if self.config.use_mellowmax:
#                         label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
#                         loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
#                     else:
#                         loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")


                    # Compute probabilities assigned to the correct labels
                    probabilities = torch.softmax(shift_logits, dim=-1)  # Shape: (batch_size, seq_len, vocab_size)
                    correct_probs = probabilities.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, seq_len)
                    negative_probs = -correct_probs  # Shape: (batch_size, seq_len)

                    # Compute predictions and matches
                    predictions = torch.argmax(shift_logits, dim=-1)  # Shape: (batch_size, seq_len)
                    matches = predictions == shift_labels  # Shape: (batch_size, seq_len)

                    # Compute the reward based on consecutive matches
                    matched_reward = -1 * (torch.cumprod(matches.float(), dim=1).sum(dim=1))

#                     # Find the index of the first mismatch for each sample
#                     seq_len = matches.size(1)
#                     indices = torch.arange(seq_len, device=matches.device).unsqueeze(0).expand_as(matches)  # Shape: (batch_size, seq_len)
#                     first_mismatch_indices = torch.where(
#                         ~matches, indices, torch.full_like(indices, seq_len)
#                     ).min(dim=1)[0]  # Shape: (batch_size,)

#                     batch_indices = torch.arange(current_batch_size, device=negative_probs.device)

#                     # Handle cases where first_mismatch_indices == seq_len to avoid indexing errors
#                     no_mismatch_mask = first_mismatch_indices == seq_len  # Shape: (batch_size,)
#                     valid_indices = first_mismatch_indices.clone()
#                     valid_indices[no_mismatch_mask] = 0  # Set to 0 or any valid index to prevent out-of-bounds

#                     # Gather negative probabilities at the first mismatch indices
#                     neg_prob_at_first_mismatch = negative_probs[batch_indices, valid_indices]  # Shape: (batch_size,)
#                     neg_prob_mean = negative_probs.mean(dim=1)  # Shape: (batch_size,)

#                     # Compute the final loss using torch.where to handle both cases
#                     loss = torch.where(
#                         no_mismatch_mask,
#                         matched_reward + neg_prob_mean,  # If no mismatches, use mean negative prob, neg_prob_mean
#                         matched_reward + neg_prob_at_first_mismatch  # If mismatches, use negative prob at first mismatch
#                     )  # Shape: (batch_size,)

                    # Compute the index of the first mismatch for each sample
                    seq_len = matches.size(1)
                    indices = torch.arange(seq_len, device=matches.device).unsqueeze(0).expand_as(matches)  # Shape: (batch_size, seq_len)
                    first_mismatch_indices = torch.where(
                        ~matches, indices, torch.full_like(indices, seq_len)
                    ).min(dim=1)[0]  # Shape: (batch_size,)

                    # Create a mask up to and including the first mismatched position
                    mask = indices <= first_mismatch_indices.unsqueeze(1)  # Shape: (batch_size, seq_len)

                    # Compute the mean of negative_probs up to and including the first mismatched position
                    neg_probs_masked = negative_probs * mask.float()  # Zero out positions beyond first mismatch
                    sum_neg_probs_masked = neg_probs_masked.sum(dim=1)  # Shape: (batch_size,)
                    mask_sum = mask.float().sum(dim=1)  # Shape: (batch_size,)
                    mean_neg_probs_before_first_mismatch = sum_neg_probs_masked / mask_sum  # Shape: (batch_size,)

                    # Compute the final loss
                    loss = matched_reward + mean_neg_probs_before_first_mismatch  # Shape: (batch_size,)
                    loss = loss * active_mask.float()
                    
                    # Append the final loss to the list of all losses
                    current_loss += loss / (seq_len + 1) * (self.config.weight ** p)

        #                 loss = loss.view(current_batch_size, -1).mean(dim=-1) + consecutive_matches
        #                 loss = custom_loss(loss.view(current_batch_size, -1))

        #                 if self.config.early_stop and not self.config.inverse:
        #                     if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
        #                         self.stop_flag = True
        
                    if not self.config.sequential:
                        active_mask = active_mask & (mask_sum == seq_len)
                        if not active_mask.any():
                            break
                    
                del outputs
                gc.collect()
                torch.cuda.empty_cache()
                all_loss.append(current_loss)
                
        if self.config.inverse:
            return -torch.cat(all_loss, dim=0)
        else:
            return torch.cat(all_loss, dim=0)

    @torch.no_grad()
    def _update_target_and_context(self, optim_str):

        # Build the input_ids
        model = self.model
        tokenizer = self.tokenizer
        
        embedding_layer = self.embedding_layer
        
        for k, template in enumerate(self.templates):
            matched_location = 0
            
            # Generate the assistant's response
            outputs, decoding_string = self.greedy_generation(template, optim_str, k)
            # Get the generated tokens and scores
            generated_ids = outputs.generated_ids  # Exclude the input_ids
            logits = outputs.scores  # List of tensors

            # Convert logits to log probabilities
            probs_list = [logits_step.softmax(dim=-1)[0] for logits_step in logits]
            num_generated_tokens = len(generated_ids)
                
            # Step 1: Collect all possible intervals
            intervals = []
            for i in range(num_generated_tokens):
                for target_text in self.targets[k]:
                    # Get the preceding tokens
                    preceding_ids = generated_ids[:i]
                    preceding_text = tokenizer.decode(preceding_ids, skip_special_tokens=True)
                    
                    next_token = tokenizer.convert_ids_to_tokens(generated_ids[i])
                    if self.config.add_space_before_target and (next_token.startswith('Ġ') or next_token.startswith(' ')):
                        combined_text = preceding_text + ' ' + target_text
                    else:
                        combined_text = preceding_text + target_text
                    combined_ids = tokenizer.encode(combined_text, add_special_tokens=False)

                    # Determine if the target string starts at position i
                    # The length of combined_ids should be equal to len(preceding_ids) + len(target_ids_in_context)
                    differences = sum(1 for x, y in zip(combined_ids[:i], preceding_ids) if x != y)

                    target_ids_in_context = combined_ids[i - differences:]
                    target_length = len(target_ids_in_context)

                    current_num_matched = 0
                    current_prob = []
                    for j in range(min(target_length, num_generated_tokens + differences - i)):
                        target_id = target_ids_in_context[j]
                        current_prob.append(probs_list[i + j - differences][target_id].item())
                        current_num_matched += 1
                        if probs_list[i + j - differences].argmax().item() != target_id:
                            current_num_matched -= 1
                            break
                            
                    current_prob = sum(current_prob) / len(current_prob)
                    current_score = (current_num_matched + current_prob) / (target_length + 1)
                    
                    if current_score >= (target_length / (target_length + 1)):
                        matched_location += 1
                        
                    start_pos = i - differences
                    end_pos = start_pos + target_length
                    intervals.append({
                        'start': start_pos,
                        'end': end_pos,
                        'score': current_score,
                        'target_ids': target_ids_in_context
                    })
                    
                if self.config.gcg:
                    break
                    
            # Step 2: Implement the Weighted Interval Scheduling Algorithm
            # Sort intervals by end position
            num_location = self.config.num_location
            intervals.sort(key=lambda x: x['end'])

            # Compute p[j] for each interval
            p = []
            for j in range(len(intervals)):
                p_j = None
                for i in range(j - 1, -1, -1):
                    if intervals[i]['end'] <= intervals[j]['start']:
                        p_j = i
                        break
                p.append(p_j)

            # Initialize DP table M[j][l]
            n = len(intervals)
            M = [[0] * (num_location + 1) for _ in range(n + 1)]

            # Fill DP table
            for j in range(1, n + 1):
                for l in range(1, num_location + 1):
                    interval = intervals[j - 1]
                    if p[j - 1] is not None:
                        include_score = interval['score'] + M[p[j - 1] + 1][l - 1]
                    else:
                        include_score = interval['score']
                    M[j][l] = max(M[j - 1][l], include_score)

            # Step 3: Reconstruct the solution
            def find_solution_iterative(M, intervals, p, j, l):
                selected = []
                while j > 0 and l > 0:
                    interval = intervals[j - 1]
                    if p[j - 1] is not None:
                        include_score = interval['score'] + M[p[j - 1] + 1][l - 1]
                    else:
                        include_score = interval['score']

                    # Check if the current interval was included in the optimal solution
                    if M[j][l] == include_score:
                        selected.append(j - 1)
                        j = p[j - 1] + 1 if p[j - 1] is not None else 0
                        l -= 1
                    else:
                        j -= 1
                return selected[::-1]  # Reverse to maintain the correct order

            selected_indices = sorted(find_solution_iterative(M, intervals, p, n, num_location))

            # Step 4: Construct the final token list
            if self.config.sequential:
                filtered_selected_indices = []
                max_score = float('-inf')
                max_score_index = None

                for idx in selected_indices:
                    interval = intervals[idx]
                    target_length = len(interval['target_ids'])
                    if interval['score'] >= (target_length / (target_length + 1)):
                        filtered_selected_indices.append(idx)
                    # Track the index with the maximum score
                    elif interval['score'] > max_score:
                        max_score = interval['score']
                        max_score_index = idx

                # If filtered_selected_indices is empty, append the index with the maximum score
                if max_score_index != None:
                    filtered_selected_indices.append(max_score_index)
                selected_indices = sorted(filtered_selected_indices)
            
            final_generated_ids = []
            prev_end = 0
            for idx in selected_indices:
                interval = intervals[idx]
                # Add tokens before the interval
                final_generated_ids.extend([
                    torch.tensor([generated_ids[prev_end:interval['start']]], device=model.device, dtype=torch.int64),
                    torch.tensor([interval['target_ids']], device=model.device, dtype=torch.int64)
                ])
                prev_end = interval['end']
            # greedy version
#             num_location = min(matched_location+1, self.config.num_location)
#             intervals.sort(key=lambda x: x['score'], reverse=True)

#             selected_intervals = []
#             for interval in intervals:
#                 # Check if this interval overlaps with any already selected interval
#                 # Overlap means: current.start < selected.end AND current.end > selected.start
#                 has_overlap = any(
#                     not (interval['end'] <= sel['start'] or interval['start'] >= sel['end']) 
#                     for sel in selected_intervals
#                 )
#                 if has_overlap:
#                     continue

#                 # If no overlap, select this interval
#                 selected_intervals.append(interval)

#                 # Stop if we've reached our limit
#                 if len(selected_intervals) == num_location:
#                     break

#             # Sort by start so we can reconstruct the final token list in the correct order
#             selected_intervals.sort(key=lambda x: x['start'])
#             final_generated_ids = []
#             prev_end = 0
#             for interval in selected_intervals:
#                 # Add tokens before the interval
#                 final_generated_ids.extend([
#                     torch.tensor([generated_ids[prev_end:interval['start']]], device=model.device, dtype=torch.int64),
#                     torch.tensor([interval['target_ids']], device=model.device, dtype=torch.int64)
#                 ])
#                 prev_end = interval['end']
            
            ## old version
#             best_score = float("-inf")
#             for i in range(num_generated_tokens):
#                 for target_text in self.targets[k]:
#                     preceding_ids = generated_ids[:i]
#                     preceding_text = tokenizer.decode(preceding_ids, skip_special_tokens=True)
                   
#                     # Re-tokenize the target string in the context of preceding text
#                     if i == 0 or ends_with_no_space_symbol(preceding_text):
#                         combined_text = preceding_text + target_text
#                     else:
#                         combined_text = preceding_text + ' ' + target_text
                    
#                     combined_ids = tokenizer.encode(combined_text, add_special_tokens=False)

#                     # Determine if the target string starts at position i
#                     # The length of combined_ids should be equal to len(preceding_ids) + len(target_ids_in_context)
#                     differences = sum(1 for x, y in zip(combined_ids[:i], preceding_ids) if x != y)

#                     target_ids_in_context = combined_ids[i - differences:]
#                     target_length = len(target_ids_in_context)

#                     current_num_matched = 0
# #                     current_prob = 0
#                     for j in range(min(target_length, num_generated_tokens + differences - i)):
#                         target_id = target_ids_in_context[j]
#                         current_prob = log_probs_list[i + j - differences][target_id].exp().item()
#                         current_num_matched += 1
#                         if log_probs_list[i + j - differences].argmax().item() != target_id:
#                             current_num_matched -= 1
#                             break
                    
#                     current_score = (current_num_matched + current_prob) / (target_length + 1) 
#                     if current_score > best_score:
#                         best_score = current_score
#                         best_pos = i - differences
#                         best_target_ids = target_ids_in_context
            
#             assistant_response_ids = torch.tensor([generated_ids[:best_pos]], device=model.device, dtype=torch.int64)
#             final_generated_ids = [torch.tensor([best_target_ids], device=model.device, dtype=torch.int64)]
#             later_response = tokenizer.decode(assistant_response_ids[0], skip_special_tokens=True)
            
#             all_response_ids = torch.cat(final_generated_ids, dim = -1)
#             assistant_response_ids = final_generated_ids.pop(0)
#             later_response = tokenizer.decode(all_response_ids[0], skip_special_tokens=True)
            
#             Get the assistant's response up to best_pos

            all_response_ids = torch.cat(final_generated_ids, dim = -1)
            assistant_response_ids = final_generated_ids.pop(0)
            later_response = tokenizer.decode(all_response_ids[0], skip_special_tokens=True)
            print(later_response)
            print("**Target**:", self.targets[0][0])
            self.batch_all_generation[k].append((optim_str, decoding_string, later_response))
            
            # Build the template and tokenize
            before_str, after_str = template.split("{optim_str}")

            # Update before_ids, after_ids, before_embeds, after_embeds
            # skip the first space
            before_ids = tokenizer([before_str.rstrip()], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(model.device).to(torch.int64)
            after_ids = tokenizer([after_str.lstrip()], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(model.device).to(torch.int64)
            after_ids = torch.cat((after_ids, assistant_response_ids), dim=-1)
            
            # Embed everything that doesn't get optimized
            before_embeds, after_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids)]
            target_embeds = [embedding_layer(ids) for ids in final_generated_ids]
            
            # Recompute prefix cache if needed
            if self.config.use_prefix_cache and self.batch_before_embeds[k] == []:
                output = model(inputs_embeds=before_embeds, use_cache=True)
                prefix_cache = output.past_key_values
                self.batch_before_ids[k]=before_ids
                self.batch_before_embeds[k]=before_embeds
                self.batch_prefix_cache[k]=prefix_cache
                
            self.batch_after_ids[k]=after_ids
            self.batch_target_ids[k]=final_generated_ids
            self.batch_after_embeds[k]=after_embeds
            self.batch_target_embeds[k]=target_embeds
            
            del outputs
            gc.collect()
            torch.cuda.empty_cache()
        
    @torch.no_grad()
    def greedy_generation(self, template, optim_str, idx=None):
        # Tokenize the messages
        input_ids = self.tokenizer([template.replace('{optim_str}', optim_str)], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(self.model.device).to(torch.int64)
        attn_masks = torch.ones_like(input_ids).to(self.model.device)
        
        # Generate the assistant's response
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_masks, 
            top_k=0, 
            top_p=1.0,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
            output_attentions=False,
            pad_token_id=self.tokenizer.eos_token_id,
            past_key_values=self.batch_prefix_cache[idx] if idx and self.batch_prefix_cache[idx] != [] else None,
            max_new_tokens=self.config.max_new_tokens,  # Adjust as needed
        )
        outputs.generated_ids = outputs.sequences[0, input_ids.shape[1]:].tolist()
        if self.tokenizer.eos_token_id and outputs.generated_ids[-1] == self.tokenizer.eos_token_id:
            outputs.generated_ids = outputs.generated_ids[:-1]
        
        decoded_string = self.tokenizer.decode(outputs.generated_ids, skip_special_tokens=False)
        ## only work for single prompt optimization ##
        if idx != None and self.config.early_stop:
            if self.config.inverse:
                if all(tgt_text not in decoded_string for tgt_text in self.targets[idx]):
                    self.stop_flag = True
            elif any((f'click[{tgt_text}]' if 'click' not in tgt_text else tgt_text) in decoded_string for tgt_text in self.targets[idx]):
                    self.stop_flag = True
        return outputs, decoded_string
    
# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict]],
    target: str,
    config: Optional[GCGConfig] = None, 
) -> GCGResult:
    """Generates a single optimized string using GCG. 

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The GCG configuration to use.
    
    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()
    
    logger.setLevel(getattr(logging, config.verbosity))
    
    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages, target)
    return result
    