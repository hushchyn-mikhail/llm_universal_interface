"""Conditional likelihood of complete class labels for a causal language model."""

import torch


def encode_candidates(tokenizer, prompts, labels, max_length):
    """Keep every query token and reject ambiguous or overlong inputs."""
    if not prompts or len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Provide at least one prompt and two distinct class labels.")
    encoded = []
    for prompt in prompts:
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        if not prefix:
            raise ValueError("Empty tokenized prompt.")
        candidates = []
        for label in labels:
            ids = tokenizer.encode(prompt + label, add_special_tokens=False)
            if ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
                raise ValueError("Label changes the prompt token boundary; check the chat template.")
            if len(ids) > max_length:
                raise ValueError(
                    f"Prompt plus label needs {len(ids)} tokens; limit is {max_length}. "
                    "Increase eval_max_seq_length within the model context limit, or explicitly "
                    "choose fewer demonstrations. No tokens were truncated."
                )
            candidates.append((ids, len(ids) - len(prefix)))
        if len({tuple(ids) for ids, _ in candidates}) != len(labels):
            raise ValueError("Distinct labels must produce distinct token sequences.")
        encoded.append(candidates)
    return encoded


@torch.inference_mode()
def class_probabilities(model, tokenizer, prompts, labels, device, max_length):
    """Softmax of summed continuation log probabilities, without a leading space.

    Each class is scored in a separate forward pass. Only the final label logits
    are materialized; this avoids a full context-by-vocabulary logits tensor.
    Scores are normalized over the supplied labels and are not calibrated
    posterior probabilities. No generation, parsing, or length normalization.
    """
    candidates = encode_candidates(tokenizer, prompts, labels, max_length)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must have a pad token.")
    scores = torch.empty((len(prompts), len(labels)), dtype=torch.float32, device=device)
    for class_index in range(len(labels)):
        batch = [row[class_index] for row in candidates]
        width = max(len(ids) for ids, _ in batch)
        keep = max(length for _, length in batch) + 1
        input_ids = torch.full(
            (len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, (ids, _) in enumerate(batch):
            input_ids[i, -len(ids):] = torch.tensor(ids, device=device)
            attention_mask[i, -len(ids):] = 1
        position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
        output = model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False, logits_to_keep=keep,
        )
        for i, (ids, length) in enumerate(batch):
            # A causal logit predicts the following token.
            logits = output.logits[i, -(length + 1):-1].float()
            target = torch.tensor(ids[-length:], dtype=torch.long, device=device)
            scores[i, class_index] = logits.log_softmax(-1).gather(
                1, target[:, None]
            ).sum()
        del output
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite class scores; check model precision and inputs.")
    return scores.softmax(-1).cpu().numpy()
