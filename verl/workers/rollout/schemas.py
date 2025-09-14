# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import os
from enum import Enum
from typing import Any, List, Optional

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from transformers import (PreTrainedTokenizer, PreTrainedTokenizerFast,
                          ProcessorMixin)

from verl.tools.schemas import (OpenAIFunctionToolCall,
                                OpenAIFunctionToolSchema, ToolResponse)
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str | dict[str, Any] | list[dict[str, Any]] | ToolResponse
    tool_calls: Optional[list[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"
    INTERACTING = "interacting"


class AsyncRolloutRequest(BaseModel):
    """The data model for async rollout.
    MODIFIED: This class is heavily refactored to use direct token ID concatenation
    instead of relying on `apply_chat_template` for multi-turn conversations,
    mimicking the vLLM implementation for better consistency and correctness.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    edit_times: int = 0
    is_bad_trajectory: bool = False
    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum
    messages: list[Message]
    multi_modal_keys: Optional[list[str]] = None
    multi_modal_data: Optional[dict[str, Any]] = None
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    tool_schemas: Optional[list[OpenAIFunctionToolSchema]] = None
    tools_kwargs: dict[str, Any] = {}
    interaction_kwargs: dict[str, Any] = {}
    input_ids: torch.Tensor
    prompt_ids: torch.Tensor
    response_ids: Optional[torch.Tensor] = None
    attention_mask: torch.Tensor
    prompt_attention_mask: torch.Tensor
    response_attention_mask: Optional[torch.Tensor] = None
    position_ids: torch.Tensor
    prompt_position_ids: torch.Tensor
    response_position_ids: Optional[torch.Tensor] = None
    loss_mask: torch.Tensor
    prompt_loss_mask: torch.Tensor
    response_loss_mask: Optional[torch.Tensor] = None
    reward_scores: dict[str, float]
    max_prompt_len: int
    max_response_len: int
    max_model_len: int
    metrics: dict[str, list[Any]] = {}
    output_token_ids: torch.Tensor | None = None
    rollout_log_probs: torch.Tensor | None = None

    # MODIFIED: Added fields for pre-encoded Qwen3 tokens.
    enable_qwen3_thinking_in_multiturn: bool
    enable_turn_reminder: bool
    assistant_gen_ids: torch.Tensor
    user_start_ids: torch.Tensor
    user_end_ids: torch.Tensor
    tool_response_start_ids: torch.Tensor
    tool_response_end_ids: torch.Tensor
    
    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        """
        MODIFIED: This validator is completely rewritten. It no longer uses `apply_chat_template`.
        It takes the `input_ids` and `attention_mask` provided from the dataloader as the ground truth
        for the initial prompt. It then pre-encodes the special Qwen3 tokens needed for subsequent turns.
        """
        if (input_ids := values.get("input_ids")) is None:
            raise ValueError("`input_ids` must be provided for AsyncRolloutRequest initialization.")
        if (attention_mask := values.get("attention_mask")) is None:
            raise ValueError("`attention_mask` must be provided for AsyncRolloutRequest initialization.")
        if not (processing_class := values.pop("processing_class", None)):
            raise ValueError("`processing_class` is required for AsyncRolloutRequest initialization.")

        # The provided input_ids and attention_mask are the initial state.
        values["prompt_ids"] = input_ids
        values["prompt_attention_mask"] = attention_mask
        
        # Initialize multi-modal data structures
        if not values.get("multi_modal_keys"):
            values["multi_modal_keys"] = ["image", "video"]
        if not values.get("multi_modal_data"):
            values["multi_modal_data"] = {key: [] for key in values["multi_modal_keys"]}
        if not values.get("multi_modal_inputs"):
            values["multi_modal_inputs"] = {}

        # Position IDs are computed from the attention mask.
        values["position_ids"] = values["prompt_position_ids"] = cls._get_position_ids(
            processing_class, input_ids, attention_mask, values["multi_modal_inputs"]
        )

        # Loss mask for the initial prompt is all False (0).
        values["loss_mask"] = values["prompt_loss_mask"] = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # MODIFIED: Pre-encode special tokens for Qwen3 agent format.
        if values.get("enable_qwen3_thinking_in_multiturn", True):
            gen_str = "\n<|im_start|>assistant\n"
        else:
            # Fallback to older format if needed
            gen_str = "\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        values["assistant_gen_ids"] = torch.tensor(processing_class.encode(gen_str, add_special_tokens=False), dtype=torch.long)
        values["user_start_ids"] = torch.tensor(processing_class.encode("\n<|im_start|>user\n", add_special_tokens=False), dtype=torch.long)
        values["user_end_ids"] = torch.tensor(processing_class.encode("<|im_end|>", add_special_tokens=False), dtype=torch.long)
        values["tool_response_start_ids"] = torch.tensor(processing_class.encode("<tool_response>\n", add_special_tokens=False), dtype=torch.long)
        values["tool_response_end_ids"] = torch.tensor(processing_class.encode("\n</tool_response>", add_special_tokens=False), dtype=torch.long)

        return values

    @staticmethod
    def _get_position_ids(
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # special case for qwen2vl
        is_qwen2vl = (
            hasattr(processing_class, "image_processor")
            and "Qwen2VLImageProcessor" in processing_class.image_processor.__class__.__name__
        )
        if is_qwen2vl:
            from verl.models.transformers.qwen2_vl import get_rope_index

            image_grid_thw = video_grid_thw = second_per_grid_ts = None
            if multi_modal_inputs:
                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

            assert input_ids.dim() == 2 and input_ids.shape[0] == 1, (
                f"input_ids should be 2D with batch size 1, but got shape {input_ids.shape}"
            )
            assert attention_mask.dim() == 2 and attention_mask.shape[0] == 1, (
                f"attention_mask should be 2D with batch size 1, but got shape {attention_mask.shape}"
            )
            new_position_ids = get_rope_index(
                processing_class,
                input_ids=input_ids.squeeze(0),
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask.squeeze(0),
            )
            return new_position_ids  # (3, seq_len)
        else:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

    def _update_input_ids(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        new_input_ids: torch.Tensor,
        attention_mask: bool,
        loss_mask: bool,
        new_multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        if new_input_ids.dim() == 1:
            new_input_ids = new_input_ids.unsqueeze(0)

        self.input_ids = torch.cat([self.input_ids, new_input_ids], dim=-1)
        
        new_attention_mask = torch.ones_like(new_input_ids) * int(attention_mask)
        self.attention_mask = torch.cat([self.attention_mask, new_attention_mask], dim=-1)
        
        new_loss_mask = torch.ones_like(new_input_ids) * int(loss_mask)
        self.loss_mask = torch.cat([self.loss_mask, new_loss_mask], dim=-1)

        if new_multi_modal_inputs:
            self._update_multi_modal_inputs(new_multi_modal_inputs)

        new_position_ids = self._get_position_ids(
            processing_class, new_input_ids, new_attention_mask, new_multi_modal_inputs
        )

        last_pos = self.position_ids[..., -1:]
        new_position_ids = new_position_ids + (last_pos + 1)

        self.position_ids = torch.cat([self.position_ids, new_position_ids], dim=-1)

        assert (
            self.input_ids.shape[-1]
            == self.attention_mask.shape[-1]
            == self.position_ids.shape[-1]
            == self.loss_mask.shape[-1]
        ), f"""Request {self.request_id} has different length of {self.input_ids.shape[-1]=}, 
            {self.attention_mask.shape[-1]=}, {self.position_ids.shape[-1]=}, {self.loss_mask.shape[-1]=}"""

    def _update_multi_modal_inputs(self, new_multi_modal_inputs: dict[str, torch.Tensor]) -> None:
        """
        Update the multi_modal_inputs of the request in additive manner.
        """
        for key in new_multi_modal_inputs:
            input_tensor = new_multi_modal_inputs[key]
            self.multi_modal_inputs[key] = (
                torch.cat([self.multi_modal_inputs[key], input_tensor], dim=0)
                if key in self.multi_modal_inputs
                else input_tensor
            )

    def get_generation_prompt_ids(self) -> list[int]:
        """
        MODIFIED: Get the generation prompt ids for rollout engine.
        This now simply returns the current full sequence of token IDs, as the
        next-turn prompt is handled explicitly by other methods.
        """
        return self.input_ids.squeeze(0).tolist()
    
    def add_user_message(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        content: str,
    ) -> None:
        """
        MODIFIED: This method now also works by encoding the string and appending IDs.
        Used for multi-turn interactions not involving tools (e.g., gsm8k).
        """
        self.messages.append(Message(role="user", content=content))
        
        # Manually construct the string and encode it
        full_str = f"\n<|im_start|>user\n{content}<|im_end|>"
        content_ids = torch.tensor(processing_class.encode(full_str, add_special_tokens=False), dtype=torch.long)
        
        self._update_input_ids(processing_class, content_ids, attention_mask=True, loss_mask=False)
        
        # Add the assistant prompt for the next turn
        self._update_input_ids(processing_class, self.assistant_gen_ids, attention_mask=True, loss_mask=False)

    def add_assistant_message(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        content: str,
        output_ids: List[int],
        tool_calls: Optional[list[OpenAIFunctionToolCall]] = None,
    ) -> None:
        """
        MODIFIED: This method now takes raw `output_ids` from the SGLang engine.
        It appends these IDs directly to the sequence tensors.
        """
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        
        new_ids = torch.tensor(output_ids, dtype=torch.long)
        # Assistant's own generation has loss enabled (loss_mask=True).
        self._update_input_ids(processing_class, new_ids, attention_mask=True, loss_mask=True)

    def add_tool_response_messages(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        contents: list[ToolResponse],
        current_turn: int,
        max_turns: int
    ) -> None:
        """
        MODIFIED: This method is completely rewritten to follow the vLLM logic.
        It constructs the tool response string, wraps it with user start/end tokens,
        encodes it, and appends the resulting IDs to the sequence.
        """
        if not contents or all(content.is_empty() for content in contents):
            # If there are no tool calls or empty responses, just prompt the assistant again.
            self._update_input_ids(processing_class, self.assistant_gen_ids, attention_mask=True, loss_mask=False)
            return

        tool_response_str = ""
        for tool_resp in contents:
            # Assuming tool_resp.text is the primary content for now.
            # Multi-modal responses from tools would need more complex handling here.
            response_text = tool_resp.text or ""
            
            tool_response_str += "<tool_response>\n"
            tool_response_str += response_text.strip() + "\n"
            if self.enable_turn_reminder:
                reminder = f"[Reminder]: You need to finish this task within {max_turns} turns. This is {current_turn + 1} turn. Remain {max_turns - current_turn - 1} turn.\n"
                tool_response_str += reminder
            tool_response_str += "</tool_response>\n"

        full_user_turn_str = f"\n<|im_start|>user\n{tool_response_str.strip()}<|im_end|>"
        
        # Encode the entire user turn (with tool responses)
        user_turn_ids = torch.tensor(processing_class.encode(full_user_turn_str, add_special_tokens=False), dtype=torch.long)
        
        # Append the user turn IDs. These are considered part of the prompt, so loss is disabled.
        self._update_input_ids(processing_class, user_turn_ids, attention_mask=True, loss_mask=False)
        
        # Now, add the prompt for the assistant's next turn, also with loss disabled.
        self._update_input_ids(processing_class, self.assistant_gen_ids, attention_mask=True, loss_mask=False)


    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def finalize(
        self,
        processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin,
        reward_scores: dict[str, list[float]],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        """
        MODIFIED: The tokenization sanity check has been completely removed as it's
        no longer relevant with the direct token ID manipulation strategy.
        """
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        
        # If the last action was an assistant generation that ended, and it was followed by a gen prompt,
        # we might need to remove it if no more turns are expected. However, the current logic adds the
        # gen prompt only when expecting a next turn, so this is generally safe.
        # Let's ensure the final sequence doesn't end with an assistant prompt if it's a terminal state.
        if finish_reason_type == FinishReasonTypeEnum.STOP or finish_reason_type == FinishReasonTypeEnum.LENGTH:
             if self.input_ids[..., -self.assistant_gen_ids.shape[-1] :].eq(self.assistant_gen_ids).all():
                self.input_ids = self.input_ids[..., : -self.assistant_gen_ids.shape[-1]]
                self.attention_mask = self.attention_mask[..., : -self.assistant_gen_ids.shape[-1]]
                self.position_ids = self.position_ids[..., : -self.assistant_gen_ids.shape[-1]]
                self.loss_mask = self.loss_mask[..., : -self.assistant_gen_ids.shape[-1]]

        self.truncate_output_ids(processing_class)

        assert (
            self.input_ids.shape[-1]
            == self.attention_mask.shape[-1]
            == self.position_ids.shape[-1]
            == self.loss_mask.shape[-1]
        ), f"""Request {self.request_id} has different length of {self.input_ids.shape[-1]=}, 
            {self.attention_mask.shape[-1]=}, {self.position_ids.shape[-1]=}, {self.loss_mask.shape[-1]=}"""

    def truncate_output_ids(
        self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
    ) -> None:
        # Truncate the whole sequence first if it exceeds max model length
        if self.input_ids.shape[-1] > self.max_model_len:
            self.is_bad_trajectory = True
            self.input_ids = self.input_ids[..., : self.max_model_len]
            self.attention_mask = self.attention_mask[..., : self.max_model_len]
            self.position_ids = self.position_ids[..., : self.max_model_len]
            self.loss_mask = self.loss_mask[..., : self.max_model_len]

        # Determine response part and truncate it if it exceeds max response length
        prompt_len = self.prompt_ids.shape[-1]
        self.response_ids = self.input_ids[..., prompt_len:]
        if self.response_ids.shape[-1] > self.max_response_len:
            self.is_bad_trajectory = True
            self.response_ids = self.response_ids[..., : self.max_response_len]
            
            # Reconstruct the full sequence with truncated response
            self.input_ids = torch.cat([self.prompt_ids, self.response_ids], dim=-1)
            total_len = self.input_ids.shape[-1]
            self.attention_mask = self.attention_mask[..., : total_len]
            self.position_ids = self.position_ids[..., : total_len]
            self.loss_mask = self.loss_mask[..., : total_len]

        # Re-slice all response-related masks
        self.response_attention_mask = self.attention_mask[..., prompt_len:]
        self.response_position_ids = self.position_ids[..., prompt_len:]
        self.response_loss_mask = self.loss_mask[..., prompt_len:]