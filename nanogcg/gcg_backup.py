import copy
import gc
import logging

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed

from nanogcg.utils import INIT_CHARS, find_executable_batch_size, get_nonascii_toks, mellowmax

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
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x"
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
    seed: int = 42
    verbosity: str = "INFO"
    tgt_freq: int = 1  # New hyperparameter
    inverse: bool = False


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    best_generation: str
    vanilla_generation: str
    vanilla_success: bool
    success: bool
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
            self.buffer.append((loss, optim_ids))
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
        n_replace: int
            the number of token positions to update per sequence
        not_allowed_ids: Tensor, shape = (n_ids)
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

def custom_loss(shift_logits, shift_labels):
    """
    Computes the custom loss per sample as per the given specifications.

    Parameters:
    - shift_logits: Tensor of shape [batch_size, seq_len, vocab_size]
    - shift_labels: Tensor of shape [batch_size, seq_len]

    Returns:
    - loss_per_sample: Tensor of shape [batch_size]
    """
    batch_size, seq_len, vocab_size = shift_logits.size()

    # Compute probabilities via softmax
    probs = torch.softmax(shift_logits, dim=-1)  # [batch_size, seq_len, vocab_size]

    # Get the probability of the target IDs
    P_target = probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len]

    # Get the maximum probability and corresponding indices
    P_max, argmax_indices = probs.max(dim=-1)  # [batch_size, seq_len]

    # Create a mask where the target ID is the argmax
    mask = (argmax_indices == shift_labels)  # [batch_size, seq_len], dtype=torch.bool

    # Convert mask to integers for cumulative product
    mask_int = mask.long()

    # Compute cumulative mask to stop processing after first False
    mask_cumulative = mask_int.cumprod(dim=1)  # [batch_size, seq_len]

    # Initialize loss contributions with -1 where mask_cumulative is 1
    loss_contributions = (-1.0) * mask_cumulative.float()  # [batch_size, seq_len]

    # Identify samples where there is any False in the mask
    mask_inv_any = (~mask).any(dim=1)  # [batch_size], dtype=torch.bool

    # Compute the first index where mask is False for each sample
    first_false_index = torch.where(
        mask_inv_any,
        (~mask).float().argmax(dim=1),
        torch.full((batch_size,), seq_len, dtype=torch.long, device=shift_logits.device)
    )  # [batch_size]

    # Adjust loss contributions at the first False index
    indices_to_adjust = (first_false_index < seq_len)
    valid_batch_indices = torch.nonzero(indices_to_adjust, as_tuple=False).squeeze(-1)
    adjust_positions = first_false_index[indices_to_adjust]

    if valid_batch_indices.numel() > 0:
        # Calculate the adjustment: -(P_target - P_max)
        loss_adjustments = -(P_target[valid_batch_indices, adjust_positions] - P_max[valid_batch_indices, adjust_positions])
        loss_contributions[valid_batch_indices, adjust_positions] = loss_adjustments

    # Sum the loss contributions over the sequence length for each sample
    loss_per_sample = loss_contributions.sum(dim=1)  # [batch_size]

    return loss_per_sample


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
        self.prefix_cache = None

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
        messages: Union[str, List[dict]],
        target: str,
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)
    
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)
    
        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"
        self.messages = messages
        
        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
            template = template.replace(tokenizer.bos_token, "")
        
        self.original_template = template
        self.target = " " + target if config.add_space_before_target else target
        self._update_target_and_context(config.optim_str_init)

        # Compute the KV Cache for tokens that appear before the optimized tokens
        if config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=self.before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values
        
        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids()

        losses = []
        optim_strings = []
        
        embedding_layer = self.embedding_layer
        for _ in tqdm(range(config.num_steps)):
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
                if self.prefix_cache:
                    input_embeds = torch.cat([
                        embedding_layer(sampled_ids),
                        self.after_embeds.repeat(new_search_width, 1, 1),
                        self.target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                else:
                    input_embeds = torch.cat([
                        self.before_embeds.repeat(new_search_width, 1, 1),
                        embedding_layer(sampled_ids),
                        self.after_embeds.repeat(new_search_width, 1, 1),
                        self.target_embeds.repeat(new_search_width, 1, 1),
                    ], dim=1)
                    
                loss = find_executable_batch_size(self.compute_candidates_loss, batch_size)(input_embeds)

                current_loss = loss.min().item()
                optim_ids = sampled_ids[loss.argmin()].unsqueeze(0)

                # Update the buffer based on the loss
                losses.append(current_loss)
                if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                    buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)
            
            buffer.log_buffer(tokenizer)                
            
            if _ % self.config.tgt_freq == 0:
                self._update_target_and_context(optim_str)
                
            if self.stop_flag:
                logger.info("Early stopping due to finding a perfect match.") 
                break
              
        min_loss_index = losses.index(min(losses)) 
        vanilla_generation=self.greedy_generation(config.optim_str_init, True)
        best_generation=self.greedy_generation(optim_strings[min_loss_index], True)
        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            best_generation= best_generation,
            success = target in best_generation,
            vanilla_generation = vanilla_generation,
            vanilla_success = target in vanilla_generation,
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
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids
                
        else: # assume list
            if (len(config.optim_str_init) != config.buffer_size):
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size) 

        # Compute the loss on the initial buffer entries
        if self.prefix_cache:
            init_buffer_embeds = torch.cat([
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)
        else:
            init_buffer_embeds = torch.cat([
                self.before_embeds.repeat(true_buffer_size, 1, 1),
                self.embedding_layer(init_buffer_ids),
                self.after_embeds.repeat(true_buffer_size, 1, 1),
                self.target_embeds.repeat(true_buffer_size, 1, 1),
            ], dim=1)

        init_buffer_losses = find_executable_batch_size(self.compute_candidates_loss, true_buffer_size)(init_buffer_embeds)

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

        if self.prefix_cache:
            input_embeds = torch.cat([optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds, past_key_values=self.prefix_cache)
        else:
            input_embeds = torch.cat([self.before_embeds, optim_embeds, self.after_embeds, self.target_embeds], dim=1)
            output = model(inputs_embeds=input_embeds)

        logits = output.logits

        # Shift logits so token n-1 predicts token n
        shift = input_embeds.shape[1] - self.target_ids.shape[1]
        shift_logits = logits[..., shift-1:-1, :].contiguous() # (1, num_target_ids, vocab_size)
        shift_labels = self.target_ids

        if self.config.use_mellowmax:
            label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
            loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
        else:
            loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optim_ids_onehot_grad = torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]
        if self.config.inverse:
            optim_ids_onehot_grad = - optim_ids_onehot_grad
        return optim_ids_onehot_grad
    
    def compute_candidates_loss(
        self,
        search_batch_size: int, 
        input_embeds: Tensor, 
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

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i+search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]
                
                if self.prefix_cache:
                    if not prefix_cache_batch or current_batch_size != search_batch_size:
                        prefix_cache_batch = [[x.expand(current_batch_size, -1, -1, -1) for x in self.prefix_cache[i]] for i in range(len(self.prefix_cache))]

                    outputs = self.model(inputs_embeds=input_embeds_batch, past_key_values=prefix_cache_batch)
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)

                logits = outputs.logits

                tmp = input_embeds.shape[1] - self.target_ids.shape[1]
                shift_logits = logits[..., tmp-1:-1, :].contiguous()
                shift_labels = self.target_ids.repeat(current_batch_size, 1)
                
                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                else:
                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
                    
                loss = loss.view(current_batch_size, -1).mean(dim=-1)
#                 loss = custom_loss(loss.view(current_batch_size, -1))
                all_loss.append(loss)

                if self.config.early_stop:
                    if torch.any(torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)).item():
                        self.stop_flag = True

                del outputs
                gc.collect()
                torch.cuda.empty_cache()
                
        if self.config.inverse:
            return -torch.cat(all_loss, dim=0)
        else:
            return torch.cat(all_loss, dim=0)

    @torch.no_grad()
    def _update_target_and_context(self, optim_str):

        # Build the input_ids
        model = self.model
        tokenizer = self.tokenizer
        
        # Generate the assistant's response
        outputs = self.greedy_generation(optim_str)

        # Get the generated tokens and scores
        generated_ids = outputs.generated_ids  # Exclude the input_ids
        logits = outputs.scores  # List of tensors

        # Convert logits to log probabilities
        log_probs_list = [logits_step.log_softmax(dim=-1)[0] for logits_step in logits]

        max_log_confidence = float('-inf')
        best_pos = 0

        num_generated_tokens = len(generated_ids)
        target_text = self.target

        for i in range(num_generated_tokens):
            # Get the preceding tokens
            preceding_ids = generated_ids[:i]
            preceding_text = tokenizer.decode(preceding_ids, skip_special_tokens=True)

            # Re-tokenize the target string in the context of preceding text
            combined_text = preceding_text + target_text
            combined_ids = tokenizer.encode(combined_text, add_special_tokens=False)

            # Determine if the target string starts at position i
            # The length of combined_ids should be equal to len(preceding_ids) + len(target_ids_in_context)
            differences = sum(1 for x, y in zip(combined_ids[:i], preceding_ids) if x != y)
            
            target_ids_in_context = combined_ids[i - differences:]
            target_length = len(target_ids_in_context)
            # Ensure we have enough generated tokens to match the target_ids_in_context
            if i + target_length - differences > num_generated_tokens:
                continue

            current_log_confidence = 0

            for j in range(target_length):
                target_id = target_ids_in_context[j]
                log_prob = log_probs_list[i + j - differences][target_id].item()
                current_log_confidence += log_prob

            if current_log_confidence > max_log_confidence:
                max_log_confidence = current_log_confidence
                best_pos = i - differences
                best_target_ids = target_ids_in_context
        
        # Now, best_pos is the position in the generated_ids where the target string starts
        # Update the target_ids to be used in the loss computation
        self.target_ids = torch.tensor([best_target_ids]).to(model.device).to(torch.int64)

        # Get the assistant's response up to best_pos
        assistant_response_ids = torch.tensor([generated_ids[:best_pos]]).to(model.device).to(torch.int64)
        assistant_response = tokenizer.decode(assistant_response_ids[0], skip_special_tokens=True)
        print("Optimial Location:", assistant_response)
        print("Target:", tokenizer.decode(best_target_ids, skip_special_tokens=True))

        # Build the template and tokenize
        template = self.original_template
        before_str, after_str = template.split("{optim_str}")

        # Update before_ids, after_ids, before_embeds, after_embeds
        self.before_ids = tokenizer([before_str], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(model.device).to(torch.int64)
        self.after_ids = tokenizer([after_str], add_special_tokens=False, padding=False, return_tensors="pt")["input_ids"].to(model.device).to(torch.int64)
        self.after_ids = torch.cat((self.after_ids, assistant_response_ids), dim=-1)
                                      
        # Embed everything that doesn't get optimized
        embedding_layer = self.embedding_layer
        self.before_embeds, self.after_embeds, self.target_embeds = [embedding_layer(ids) for ids in (self.before_ids, self.after_ids, self.target_ids)]

        # Recompute prefix cache if needed
        if self.config.use_prefix_cache:
            with torch.no_grad():
                output = model(inputs_embeds=self.before_embeds, use_cache=True)
                self.prefix_cache = output.past_key_values
        
        del outputs
        gc.collect()
        torch.cuda.empty_cache()
        
    def greedy_generation(self, optim_str, decoding=False):
        # Tokenize the messages
        input_ids = self.tokenizer([self.original_template.replace('{optim_str}', optim_str)], padding=False, return_tensors="pt")["input_ids"].to(self.model.device).to(torch.int64)
        attn_masks = torch.ones_like(input_ids).to(self.model.device)
        
        # Generate the assistant's response
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_masks, 
            top_k=0, 
            top_p=1.0,
            do_sample=False,
            max_length=4096,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=False,
            output_attentions=False,
            pad_token_id=self.tokenizer.eos_token_id
#             max_new_tokens=50,  # Adjust as needed
        )
        outputs.generated_ids = outputs.sequences[0, input_ids.shape[1]:].tolist()
        if decoding:
            outputs = self.tokenizer.decode(outputs.generated_ids, skip_special_tokens=True)
            
        return outputs
    
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
    