import re
import torch
import torch.nn.functional as F

STOP_WORDS = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "by", "for", "with", "about", "as", "into",
    "it", "its", "this", "that", "these", "those", "i", "you", "he", "she",
    "they", "them", "his", "her", "their", "my", "your", "our", "and", "or",
    "but", "if", "so", "just", "very", "really", "too", "also", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "can", "may"
}

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "none", "nothing", "nowhere",
    "dont", "don't", "doesnt", "doesn't", "didnt", "didn't", "wont", "won't",
    "cant", "can't", "cannot", "isnt", "isn't", "arent", "aren't", "wasnt",
    "wasn't", "werent", "weren't", "aint", "ain't", "hardly", "barely"
}


class SemanticAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.embed_layer = self.model.get_input_embeddings()
        self.device = self.embed_layer.weight.device

        # Compute global embedding mean for anisotropy correction (All-but-the-Top centering).
        # Raw causal LLM token embeddings lie in a narrow cone; subtracting the mean vector
        # centers the space around the origin so cosine similarity measures true semantic direction.
        print("Initializing Semantic Embedding Space...")
        with torch.no_grad():
            weights = self.embed_layer.weight.detach().to(dtype=torch.float32)
            self.embedding_mean = weights.mean(dim=0, keepdim=True).cpu()

        # We define "Anchor Coordinates" in the centered math space.
        anchor_definitions = {
            "anger": "hate kill bad threat angry stupid error fail urgent attack hurt furious rage annoying idiot",
            "joy": "good love great friend happy thanks together hug beautiful smile awesome glad sweet care",
            "focus": "why how explain think complex analyze code logic work study math debug problem solve theory",
            "sadness": "sad cry hurt pain sorry miss alone lonely tragic broken died loss dead grief depressed hopeless",
            "hunger": "hungry eat food starving snack craving pizza meal delicious burger famished appetite thirsty",
            "lazy": "full ate relax chill done lazy comfortable rest peaceful calm content couch easy unbothered",
            "motivation": "want desire yes win amazing excite goal achieve succeed push victory strong improve hype",
            "sleep": "tired sleep night boring exhausted yawn bed snooze nap fatigue sleepy slumber pillow weary"
        }

        self.anchors = {
            emotion: self.get_embedding(text, apply_weights=False)
            for emotion, text in anchor_definitions.items()
        }
        self.anchor_names = list(self.anchors.keys())
        # Stack all 8 normalized (1, D) anchor vectors into a single (8, D) matrix for vectorized scoring
        self.anchor_matrix = torch.cat([self.anchors[k] for k in self.anchor_names], dim=0)

    def get_embedding(self, text: str, apply_weights: bool = True) -> torch.Tensor:
        """
        Takes a sentence, passes it through the LLM's raw embedding layer (bypassing the heavy brain),
        applies anisotropy mean-centering and token-importance weighting, and returns its normalized vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids
        if input_ids.numel() == 0:
            dim = self.embedding_mean.shape[-1]
            return torch.zeros((1, dim), dtype=torch.float32)

        with torch.no_grad():
            raw_embeds = self.embed_layer(input_ids.to(self.device)).to(device="cpu", dtype=torch.float32)
            # 1. Subtract global mean to eliminate embedding cone anisotropy
            centered = raw_embeds - self.embedding_mean.unsqueeze(0)

            if not apply_weights or input_ids.shape[1] <= 1:
                sentence_vector = centered.mean(dim=1)
                return F.normalize(sentence_vector, p=2, dim=1)

            # 2. Norm-based information weighting + stop-word & negation handling
            tokens = [
                re.sub(r"^[^a-zA-Z0-9']+|[^a-zA-Z0-9']+$", "", self.tokenizer.decode([tid]).strip().lower())
                for tid in input_ids[0].tolist()
            ]
            token_norms = torch.norm(centered[0], p=2, dim=-1)
            weights = token_norms.clone()

            for idx, tok in enumerate(tokens):
                if not tok or tok in STOP_WORDS:
                    weights[idx] *= 0.15
                elif tok in NEGATION_WORDS:
                    weights[idx] *= 0.10
                else:
                    # Check if preceded by a negation token within a 2-token window
                    prev1 = tokens[idx - 1] if idx >= 1 else ""
                    prev2 = tokens[idx - 2] if idx >= 2 else ""
                    if prev1 in NEGATION_WORDS or prev2 in NEGATION_WORDS:
                        # Invert and dampen negated content tokens so "not happy" doesn't match "joy"
                        weights[idx] *= -0.5

            weight_sum = weights.abs().sum().clamp(min=1e-6)
            sentence_vector = (centered[0] * weights.unsqueeze(-1)).sum(dim=0, keepdim=True) / weight_sum
            return F.normalize(sentence_vector, p=2, dim=1)

    def analyze(self, user_input: str):
        """
        Calculates how close the user's sentence is to our emotional anchors using a single vectorized matmul.
        """
        if not user_input.strip():
            return {k: 0.0 for k in self.anchor_names}

        user_vec = self.get_embedding(user_input, apply_weights=True)
        # Single (1, D) @ (D, 8) matrix multiplication instead of 8 separate torch.mm calls
        similarities = torch.mm(user_vec, self.anchor_matrix.T).squeeze(0).tolist()
        return dict(zip(self.anchor_names, similarities))


# --- Quick Test Block ---
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
    print("Loading test model...")
    model_id = "huihui-ai/Huihui-Qwen3.5-35B-A3B-abliterated"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

    analyzer = SemanticAnalyzer(model, tokenizer)

    test_sentence = "my parents died"
    print(f"\nTesting sentence: '{test_sentence}'")
    results = analyzer.analyze(test_sentence)

    # Sort and print the closest emotional matches
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for emotion, score in sorted_results:
        print(f"{emotion}: {score:.3f}")
