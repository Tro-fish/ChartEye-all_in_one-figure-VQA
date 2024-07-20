import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
from collections import defaultdict
from pycocoevalcap.cider.cider import Cider

# 평가 데이터 파일 경로 설정
json_file_path = '../dataset/eval/validation.json'

# BLEU4 평가 함수
def calculate_bleu4(predictions, references):
    smoothing_function = SmoothingFunction().method4
    scores = []
    for pred, ref in zip(predictions, references):
        score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing_function)
        scores.append(score)
    return np.mean(scores) * 100

# ROUGE 평가 함수
def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = defaultdict(list)
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in score:
            scores[key].append(score[key].fmeasure)
    return {key: np.mean([v * 100 for v in val]) for key, val in scores.items()}

# METEOR 평가 함수
def calculate_meteor(predictions, references):
    scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        score = meteor_score([ref_tokens], pred_tokens)
        scores.append(score)
    return np.mean(scores)*100

# CIDEr 평가 함수
def calculate_cider(predictions, references):
    cider_scorer = Cider()
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [pred] for i, pred in enumerate(predictions)}
    cider_score, _ = cider_scorer.compute_score(gts, res)
    return cider_score*10

# JSON 파일 읽기 및 평가
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 초기화
bleu4_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
meteor_scores = []

# CIDEr 점수를 위한 데이터 수집
cider_predictions = []
cider_references = []

# 데이터 처리 및 평가
for item in tqdm(data, total=len(data)):
    pred = item['predition'].lower()
    ref = item['gold'].lower()

    pred_tokens = pred.split()
    ref_tokens = ref.split()

    # BLEU4
    bleu4_score = calculate_bleu4([pred], [ref])
    bleu4_scores.append(bleu4_score)

    # ROUGE
    rouge_score = calculate_rouge([pred], [ref])
    rouge1_scores.append(rouge_score['rouge1'])
    rouge2_scores.append(rouge_score['rouge2'])
    rougeL_scores.append(rouge_score['rougeL'])

    # METEOR
    meteor_score_value = calculate_meteor([pred], [ref])
    meteor_scores.append(meteor_score_value)

    # CIDEr
    cider_predictions.append(pred)
    cider_references.append(ref)

# CIDEr 점수 계산
cider_score_value = calculate_cider(cider_predictions, cider_references)

# 최종 평균 점수 계산
average_bleu4 = np.mean(bleu4_scores)
average_rouge1 = np.mean(rouge1_scores)
average_rouge2 = np.mean(rouge2_scores)
average_rougeL = np.mean(rougeL_scores)
average_meteor = np.mean(meteor_scores)

# 결과 출력
print(f"Average BLEU-4 Score: {average_bleu4}")
print(f"Average ROUGE-1 Score: {average_rouge1}")
print(f"Average ROUGE-2 Score: {average_rouge2}")
print(f"Average ROUGE-L Score: {average_rougeL}")
print(f"Average METEOR Score: {average_meteor}")
print(f"Average CIDEr Score: {cider_score_value}")