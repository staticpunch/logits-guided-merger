from transformers import AutoTokenizer
import json
import os
from difflib import unified_diff

def are_tokenizers_identical(tokenizer1_path, tokenizer2_path):
    """
    Compares two Hugging Face tokenizers to determine if they are completely identical.

    Args:
        tokenizer1_path (str): Path to the directory of the first tokenizer or its identifier on the Hugging Face Hub.
        tokenizer2_path (str): Path to the directory of the second tokenizer or its identifier on the Hugging Face Hub.

    Returns:
        tuple: (bool, list)
            - bool: True if the tokenizers are identical, False otherwise.
            - list: A list of strings describing the differences found, empty if tokenizers are identical.
    """
    diffs = []

    try:
        tokenizer1 = AutoTokenizer.from_pretrained(tokenizer1_path)
        tokenizer2 = AutoTokenizer.from_pretrained(tokenizer2_path)
    except Exception as e:
        diffs.append(f"Error loading tokenizers: {e}")
        return False, diffs

    # 1. Compare Tokenizer Class
    if tokenizer1.__class__.__name__ != tokenizer2.__class__.__name__:
        diffs.append(f"Tokenizer classes differ: {tokenizer1.__class__.__name__} vs {tokenizer2.__class__.__name__}")

    # 2. Compare Config Files
    def compare_json_files(file1, file2, filename):
        nonlocal diffs
        try:
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                data1 = json.load(f1)
                data2 = json.load(f2)
                if data1 != data2:
                    diffs.append(f"Differences found in {filename}:")
                    for line in unified_diff(
                        json.dumps(data1, indent=2).splitlines(keepends=True),
                        json.dumps(data2, indent=2).splitlines(keepends=True),
                        fromfile=file1,
                        tofile=file2,
                    ):
                        diffs.append(line.strip())
        except FileNotFoundError:
            diffs.append(f"One or both {filename} files are missing.")
        except json.JSONDecodeError:
            diffs.append(f"One or both {filename} files are not valid JSON.")

    config_file1 = os.path.join(tokenizer1_path, "tokenizer_config.json")
    config_file2 = os.path.join(tokenizer2_path, "tokenizer_config.json")
    compare_json_files(config_file1, config_file2, "tokenizer_config.json")

    special_tokens_map_file1 = os.path.join(tokenizer1_path, "special_tokens_map.json")
    special_tokens_map_file2 = os.path.join(tokenizer2_path, "special_tokens_map.json")
    compare_json_files(special_tokens_map_file1, special_tokens_map_file2, "special_tokens_map.json")

    # 3. Compare Vocabulary Files
    def load_and_compare_vocab_files(filepath1, filepath2, filename, load_func):
        nonlocal diffs
        try:
            vocab1 = load_func(filepath1)
            vocab2 = load_func(filepath2)

            if vocab1 != vocab2:
                diffs.append(f"Differences found in {filename}:")

                if isinstance(vocab1, list) and isinstance(vocab2, list):
                  set1 = set(vocab1)
                  set2 = set(vocab2)
                  diffs.append(f"  Items in {filepath1} but not in {filepath2}: {set1 - set2}")
                  diffs.append(f"  Items in {filepath2} but not in {filepath1}: {set2 - set1}")

                  for i in range(min(len(vocab1), len(vocab2))):
                      if vocab1[i] != vocab2[i]:
                        diffs.append(f"  Difference at index {i}: '{vocab1[i]}' vs '{vocab2[i]}'")

                elif isinstance(vocab1, dict) and isinstance(vocab2, dict):
                    for key in set(vocab1.keys()).union(vocab2.keys()):
                        if key not in vocab1:
                            diffs.append(f"  Key '{key}' missing in {filepath1}")
                        elif key not in vocab2:
                            diffs.append(f"  Key '{key}' missing in {filepath2}")
                        elif vocab1[key] != vocab2[key]:
                            diffs.append(f"  Value mismatch for key '{key}': {vocab1[key]} vs {vocab2[key]}")

                elif isinstance(vocab1, list) and isinstance(vocab2, list):
                    for i in range(len(vocab1)):
                        if i >= len(vocab2):
                            diffs.append(f"  Missing lines starting at index {i} in {filepath2}")
                            break
                        if vocab1[i] != vocab2[i]:
                            diffs.append(f"  Difference at index {i}: {vocab1[i]} vs {vocab2[i]}")

        except FileNotFoundError:
            diffs.append(f"One or both {filename} files are missing.")

    def load_vocab_txt(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def load_vocab_json(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_merges_txt(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [tuple(line.strip().split()) for line in f if not line.startswith("#")]
    
    vocab_file1 = os.path.join(tokenizer1_path, "vocab.txt")
    vocab_file2 = os.path.join(tokenizer2_path, "vocab.txt")

    if os.path.exists(vocab_file1) and os.path.exists(vocab_file2):
      load_and_compare_vocab_files(vocab_file1, vocab_file2, "vocab.txt", load_vocab_txt)
    else:
        vocab_json_file1 = os.path.join(tokenizer1_path, "vocab.json")
        vocab_json_file2 = os.path.join(tokenizer2_path, "vocab.json")
        merges_file1 = os.path.join(tokenizer1_path, "merges.txt")
        merges_file2 = os.path.join(tokenizer2_path, "merges.txt")

        if os.path.exists(vocab_json_file1) and os.path.exists(vocab_json_file2) and os.path.exists(merges_file1) and os.path.exists(merges_file2):
          load_and_compare_vocab_files(vocab_json_file1, vocab_json_file2, "vocab.json", load_vocab_json)
          load_and_compare_vocab_files(merges_file1, merges_file2, "merges.txt", load_merges_txt)
        else:
          diffs.append("Could not find vocabulary files (vocab.txt or vocab.json/merges.txt) for comparison.")



    # 4. Compare Tokenization Output
    sample_texts = [
        "This is a simple test.",
        "Hello, world!",
        "  Extra spaces  ",
        "A sentence with punctuation!?!",
        "Unicode characters: こんにちは世界",
        "123 456 789",
        "<|endoftext|>",
        "A very long sentence that might exceed the maximum length",
        " ",
        "",
        "Sentence1\nSentence2",
        "Test with  multiple   spaces.",
    ]

    for text in sample_texts:
        try:
            encoding1 = tokenizer1(text, return_tensors="pt")
            encoding2 = tokenizer2(text, return_tensors="pt")

            if (encoding1["input_ids"] == encoding2["input_ids"]).all() and (
                encoding1["attention_mask"] == encoding2["attention_mask"]
            ).all():
                continue
            else:
                diffs.append(f"Tokenization output for '{text}' is different:")
                diffs.append(f"  Tokenizer 1: {encoding1}")
                diffs.append(f"  Tokenizer 2: {encoding2}")
        except Exception as e:
            diffs.append(f"Error during tokenization of '{text}': {e}")
    
    # 5. Compare name_or_path
    if tokenizer1.name_or_path != tokenizer2.name_or_path:
        diffs.append(
            f"Tokenizers loaded from different identifiers: {tokenizer1.name_or_path} vs {tokenizer2.name_or_path}"
        )

    return len(diffs) == 0, diffs