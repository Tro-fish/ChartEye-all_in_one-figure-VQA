import numpy as np
from scipy import optimize

# Function to parse tables from a given text format
def parse_table(text):
    lines = text.strip().split("\n")
    headers = lines[1].split(" | ")
    rows = [line.split(" | ") for line in lines[2:]]
    return headers, rows

# Function to convert text to float if possible
def _to_float(text):
    try:
        if text.endswith("%"):
            # Convert percentages to floats.
            return float(text.rstrip("%")) / 100.0
        else:
            return float(text)
    except ValueError:
        return None

# Compute Normalized Levenshtein Distance
def normalized_levenshtein_distance(s1, s2, tau=0.5):
    len_s1, len_s2 = len(s1), len(s2)
    d = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
    for i in range(len_s1 + 1):
        d[i][0] = i
    for j in range(len_s2 + 1):
        d[0][j] = j
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    distance = d[len_s1][len_s2]
    return min(distance / max(len_s1, len_s2), tau)

# Function to compute relative distance
def _get_relative_distance(target, prediction, theta=0.5):
    if target is None or prediction is None:
        return 1.0
    distance = min(abs((target - prediction) / target), 1)
    return distance if distance < theta else 1

# Function to compute RMS between two tables
def compute_rms(target_headers, target_rows, prediction_headers, prediction_rows, tau=0.1, theta=0.5):
    N = len(prediction_rows)
    M = len(target_rows)
    similarity_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            key_sim = 1 - normalized_levenshtein_distance(prediction_rows[i][0], target_rows[j][0], tau)
            value_sim = 1 - _get_relative_distance(_to_float(prediction_rows[i][1]), _to_float(target_rows[j][1]), theta)
            similarity_matrix[i, j] = key_sim * value_sim
    
    row_ind, col_ind = optimize.linear_sum_assignment(-similarity_matrix)
    total_similarity = similarity_matrix[row_ind, col_ind].sum()
    
    RMS_precision = total_similarity / N
    RMS_recall = total_similarity / M
    RMS_F1 = 2 * (RMS_precision * RMS_recall) / (RMS_precision + RMS_recall) if RMS_precision + RMS_recall > 0 else 0
    
    return RMS_precision, RMS_recall, RMS_F1

# Example data
target_table = """title | my table
year | argentina | brazil
1999 | 200 | 158
"""
prediction_table = """title | my table
time | argina | brdfil
1999 | 145 | 123
"""

# Parse tables
target_headers, target_rows = parse_table(target_table)
prediction_headers, prediction_rows = parse_table(prediction_table)

# Compute RMS
precision, recall, f1 = compute_rms(target_headers, target_rows, prediction_headers, prediction_rows)

# Output results
print(f"RMS Precision: {precision}")
print(f"RMS Recall: {recall}")
print(f"RMS F1 Score: {f1}")