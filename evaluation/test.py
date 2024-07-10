# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""Binary to compute table equality metrics."""

import csv
import json
import os
import zipfile

from absl import app
from absl import flags
import tensorflow as tf

import metrics


_JSONL = flags.DEFINE_string(
    'jsonl', None, 'JSONL directory with predictions')


def main(argv):

  _JSONL = '/home/wani/Desktop/Corning_team3/evaluation/dummy_data.jsonl'
  targets, predictions = [], []


  with tf.io.gfile.GFile(_JSONL) as f:
    for line in f:
      example = json.loads(line)
      targets.append(example['target'])
      predictions.append(example['prediction'])


  metric = {}
  metric.update(metrics.table_datapoints_precision_recall(targets, predictions))
  metric.update(metrics.table_number_accuracy(targets, predictions))
  metric_log = json.dumps(metric, indent=2)
  
  print(metric_log)

if __name__ == '__main__':
  app.run(main)