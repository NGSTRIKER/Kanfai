import torch
import torch.nn.functional as F

class SemanticAnalyzer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # We define "Anchor Coordinates" in the math space.
        # These are the perfect mathematical representations of our 8 categories.
        print("Initializing Semantic Embedding Space...")
        self.anchors = {
            "anger": self.get_embedding("hate kill bad threat angry stupid error fail urgent attack hurt"),
            "joy": self.get_embedding("good love great friend happy thanks together hug beautiful smile"),
            "focus": self.get_embedding("why how explain think complex analyze code logic work study math"),
            "sadness": self.get_embedding("sad cry hurt pain sorry miss alone lonely tragic broken died loss dead"),
            "hunger": self.get_embedding("hungry eat food starving snack craving pizza meal delicious"),
            "lazy": self.get_embedding("full ate relax chill done lazy comfortable rest peaceful calm"),
            "motivation": self.get_embedding("want desire yes win amazing excite goal achieve succeed push"),
            "sleep": self.get_embedding("tired sleep night boring exhausted yawn bed snooze nap fatigue")
        }

    def get_embedding(self, text):
        """
        Takes a sentence, passes it through the LLM's raw embedding layer (bypassing the heavy brain),
        and returns its exact 3D coordinate vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to("cpu")
        with torch.no_grad():
            # Get the raw vectors for each word
            # Note: We use model.model.embed_tokens to just use the dictionary, not the whole AI!
            outputs = self.model.model.embed_tokens(inputs.input_ids)
            
            # Average the words together to get the 'Sentence Coordinate'
            sentence_vector = outputs.mean(dim=1)
            
            # Normalize it so we can easily calculate distance
            return F.normalize(sentence_vector, p=2, dim=1)

    def analyze(self, user_input):
        """
        Calculates how close the user's sentence is to our emotional anchors.
        """
        if not user_input.strip():
            return {k: 0.0 for k in self.anchors.keys()}

        user_vec = self.get_embedding(user_input)
        scores = {}
        
        for emotion, anchor_vec in self.anchors.items():
            # Cosine Similarity: How close are these two dots on the math map? (0.0 to 1.0)
            similarity = torch.mm(user_vec, anchor_vec.T).item()
            scores[emotion] = similarity
            
        return scores

# --- Quick Test Block ---
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading test model...")
    model_id = "thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    
    analyzer = SemanticAnalyzer(model, tokenizer)
    
    test_sentence = "my parents died"
    print(f"\nTesting sentence: '{test_sentence}'")
    results = analyzer.analyze(test_sentence)
    
    # Sort and print the closest emotional matches
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for emotion, score in sorted_results:
        print(f"{emotion}: {score:.3f}")
