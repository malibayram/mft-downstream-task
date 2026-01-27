from transformers import GPT2Tokenizer
import torch

class ReversedGPT2Tokenizer(GPT2Tokenizer):
    """
    A tokenizer that inherits from GPT2Tokenizer but reverses the tokenized text.
    """
    
    def _tokenize(self, text):
        """
        Tokenize a string and reverse the order of the tokens.
        """
        # Use the original tokenizer's logic to tokenize the text
        bpe_tokens = super()._tokenize(text)
        
        # Reverse the list of tokens
        reversed_bpe_tokens = bpe_tokens[::-1]
        
        return reversed_bpe_tokens

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequences by reversing the token order
        while keeping the special tokens in their appropriate positions.
        """
        # Call the parent method to handle special tokens
        output = super().build_inputs_with_special_tokens(token_ids_0, token_ids_1)
        
        # Special tokens (e.g. [BOS], [EOS]) should remain in place, so we reverse only the content tokens
        if self.add_bos_token:
            special_token_id = output[0]
            return [special_token_id] + output[1:][::-1]
        else:
            return output[::-1]

    def encode_plus(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        **kwargs
    ):
        """
        Encode the text and reverse the order of tokens while handling special tokens.
        """
        # First tokenize the text
        encoded = super().encode_plus(text, text_pair, add_special_tokens=add_special_tokens, **kwargs)
        
        # Reverse the order of the tokens
        encoded['input_ids'] = encoded['input_ids'][::-1]
        
        return encoded
    
    def batch_encode_plus(
        self,
        batch_text_or_text_pairs,
        add_special_tokens=True,
        **kwargs
    ):
        """
        Encode a batch of text and reverse the tokenized text for each instance.
        Handles both list and tensor inputs appropriately.
        """
        encoded_batch = super().batch_encode_plus(batch_text_or_text_pairs, add_special_tokens=add_special_tokens, **kwargs)
        
        # Check if input_ids is a list or tensor and reverse accordingly
        if isinstance(encoded_batch['input_ids'], list):
            # Handle list input
            encoded_batch['input_ids'] = [input_ids[::-1] for input_ids in encoded_batch['input_ids']]
        else:
            # Handle tensor input
            if isinstance(encoded_batch['input_ids'], torch.Tensor):
                # Use torch.flip for tensor reversal along dimension 1 (sequence length)
                encoded_batch['input_ids'] = torch.flip(encoded_batch['input_ids'], dims=[1])
            else:
                raise TypeError("input_ids must be either a list or a torch.Tensor")
        
        return encoded_batch
