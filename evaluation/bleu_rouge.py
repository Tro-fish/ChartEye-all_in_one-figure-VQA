import sacrebleu
from rouge_score import rouge_scorer

# Sample predictions and references
predicted_caption = "This is a predicted caption for the chart."
reference_caption = "This is the reference caption for the chart."

# Function to calculate BLEU score using sacrebleu
def calculate_bleu(predicted, reference):
    # sacrebleu expects references to be a list of lists
    reference = [reference]
    bleu = sacrebleu.sentence_bleu(predicted, reference)
    return bleu.score

# Function to calculate ROUGE score
def calculate_rouge(predicted, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, predicted)
    return scores

# Calculate and print BLEU score
bleu_score = calculate_bleu(predicted_caption, reference_caption)
print(f"BLEU score: {bleu_score}")

# Calculate and print ROUGE score
rouge_scores = calculate_rouge(predicted_caption, reference_caption)
print(f"ROUGE scores: {rouge_scores}")