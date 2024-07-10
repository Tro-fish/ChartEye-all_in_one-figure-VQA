import numpy as np
from scipy.optimize import linear_sum_assignment

def extract_numbers(table):
    lines = table.strip().split('\n')
    numbers = []
    for line in lines:
        parts = line.split('|')
        for part in parts:
            part = part.strip()
            try:
                number = float(part)
                numbers.append(number)
            except ValueError:
                continue
    return numbers

def relative_distance(p, t):
    return min(1, abs(p - t) / abs(t))

def compute_rnss(target_table, prediction_table):
    target_numbers = extract_numbers(target_table)
    prediction_numbers = extract_numbers(prediction_table)
    
    if not target_numbers and not prediction_numbers:
        return 1.0
    if not target_numbers or not prediction_numbers:
        return 0.0
    
    N = len(prediction_numbers)
    M = len(target_numbers)
    max_len = max(N, M)
    
    cost_matrix = np.zeros((N, M))
    for i, p in enumerate(prediction_numbers):
        for j, t in enumerate(target_numbers):
            cost_matrix[i, j] = relative_distance(p, t)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    rnss = 1 - cost_matrix[row_ind, col_ind].sum() / max_len
    return rnss

# 예시 데이터
target_table = """title | my table
year | argentina | brazil
1999 | 200 | 158
"""
prediction_table = """title | my table
year | argentina | brazil
1999 | 1 | 12
"""

rnss_score = compute_rnss(target_table, prediction_table)
print("RNSS Score:", rnss_score)