import collections

from sklearn.metrics import f1_score

# A file of helper functions

def CopyCheckpointFolder(resume_from_checkpoint,output_directory):
    '''
    copy the files neede in the output directory with a name checkpoint-resume_from_checkpoint
    the folder will be delete as we only save limited checkpoint, but its exactly what we wanna
    return the new resume_from_checkpoint folder address
    '''
    import shutil
    import os    
    #set the folder adress
    new_resume_from_checkpoint=output_directory+'/checkpoint-resume_from_checkpoint'
    
    #delete the existing folder
    print('deleting the existing checkpoint-resume_from_checkpoint folder')
    os.makedirs(os.path.dirname(output_directory), exist_ok=True)
    shutil.rmtree(new_resume_from_checkpoint,ignore_errors=True)
    
    #copy the folder
    print(f'copying from {resume_from_checkpoint} into {new_resume_from_checkpoint} folder')
    os.makedirs(os.path.dirname(new_resume_from_checkpoint), exist_ok=True)
    shutil.copytree(resume_from_checkpoint,new_resume_from_checkpoint, symlinks=False, ignore=None, copy_function=shutil.copy2, ignore_dangling_symlinks=False,  dirs_exist_ok=False)
        
    #remove the optimizer, scheduler and trainer_state

    for i in ['optimizer.pt','scheduler.pt','trainer_state.json']:
        file_to_remove=f"{new_resume_from_checkpoint}/{i}"
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)

    
    return new_resume_from_checkpoint


def squad(targets, predictions):
  """Computes SQuAD metrics, maximizing over answers per question.
  Args:
    targets: list of lists of strings
    predictions: list of strings
  Returns:
    dict with score_key: squad score across all targets and predictions
  """
  targets = [[normalize_squad(t) for t in u] for u in targets]
  predictions = [normalize_squad(p) for p in predictions]
  return qa_metrics(targets, predictions)

import collections
import re
import string

from absl import logging
import numpy as np


def _normalize_answer(text, punc_chars, punc_repl):
  """Lower text and remove punctuation, articles and extra whitespace."""

  def remove_articles(s):
    return re.sub(r"\b(a|an|the)\b", " ", s)

  def replace_punctuation(s):
    to_replace = set(punc_chars)
    return "".join(punc_repl if ch in to_replace else ch for ch in s)

  def white_space_fix(s):
    return " ".join(s.split())

  text = text.lower()
  text = replace_punctuation(text)
  text = remove_articles(text)
  text = white_space_fix(text)

  return text


def normalize_trivia_qa(answer):
  """Normalization used in official TriviaQA evaluation script."""
  return _normalize_answer(
      answer, punc_chars=string.punctuation + "‘’´`_", punc_repl=" ").strip()


def normalize_squad(answer):
  """Normalization used in official SQuAD evaluation script."""
  return _normalize_answer(answer, punc_chars=string.punctuation, punc_repl="")


def _metric_max_over_ground_truths(metric_fn, ground_truths, prediction):
  """Computes the maximum of the metric over all ground truths."""
  return max(
      metric_fn(ground_truth, prediction) for ground_truth in ground_truths
  )


def _exact_match_score(target, prediction):
  return target == prediction


def _f1_score(target, prediction):
  """Computes token f1 score for a single target and prediction."""
  prediction_tokens = prediction.split()
  target_tokens = target.split()
  common = (collections.Counter(prediction_tokens) &
            collections.Counter(target_tokens))
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(target_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def qa_metrics(targets, predictions):
  """Computes exact match and f1 QA scores, expecting pre-normalized text."""
  if len(targets) != len(predictions):
    raise ValueError("Number of targets and predictions must match.")
  em = np.mean([
      _metric_max_over_ground_truths(_exact_match_score, t, p)
      for p, t in zip(predictions, targets)
  ])
  f1 = np.mean([
      _metric_max_over_ground_truths(_f1_score, t, p)
      for p, t in zip(predictions, targets)
  ])
  logging.info("EM = %.4f, F1 = %.4f", em, f1)
  return {"F1": f1,"EM": em }



# define metric for EM and mean_group 

def all_match(targets, predictions):
  """Computes whether all targets match all predictions exactly."""
  return {"EM": float(np.array_equal(targets, predictions))}

def mean_group_metric(metric_fn,
                      group_key="group_idx",
                      value_key="original_target",
                      return_subgroup_scores=False):
  """Returns a metric that averages `metric_fn` on sub-groups of results.
  The sub-groups are defined by aggregating results (targets and predictions)
  by accessing the feature specified by `group_key` in the target dicts.
  **WARNING**: Using this function can produce unreliable results if you do not
  pass in full groups. For example, if you evaluate over a random subsample of a
  validation set and do not retain all of the examples in each group, you may
  get results which aren't directly comparable to using the full validation set.
  Args:
    metric_fn: function, the metric to compute on the subgroups.
    group_key: string, the key for the grouping value in the target dictionary.
    value_key: string, the key for the value in the dictionaries.
    return_subgroup_scores: If true, include the scores for each sub-group.
  """
  def my_metric(targets, predictions):
    """Computes mean of `metric_fn` over subgroups of results."""
    grouped_values = collections.defaultdict(lambda: ([], []))
    for targ, pred in zip(targets, predictions):
        
#       print(targ,group_key,pred)
    
      g = targ[group_key]
      grouped_values[g][0].append(targ[value_key])
      grouped_values[g][1].append(pred[value_key])
    group_scores = collections.defaultdict(list)
    for group, (targets, predictions) in grouped_values.items():
      for metric, score in metric_fn(targets, predictions).items():
        group_scores[metric].append(score)
        if return_subgroup_scores:
          group_scores["%s-%s" % (group, metric)].append(score)
    return {'mean_group_'+metric: np.mean(scores) for metric, scores in group_scores.items()}
  return my_metric



def MultircFinalMetric(eval_dataset,predictions):
    predictions_list=[{'original_target':i} for i in predictions]
    
    result_dict=multirc_f1_over_all_answers(eval_dataset,predictions_list)
    
    result_dict.update(mean_group_metric(all_match)(eval_dataset,predictions_list))
    
    logging.info(result_dict)
    
    return result_dict

def f1_score_with_invalid(targets, predictions):
  """Compute F1 score, but any prediction != 0 or 1 is counted as incorrect.
  Args:
    targets: np.ndarray of targets, either 0 or 1
    predictions: np.ndarray of predictions, any integer value
  Returns:
    F1 score, where any prediction != 0 or 1 is counted as wrong.
  """
  targets, predictions = np.asarray(targets), np.asarray(predictions)
  # Get indices of invalid predictions
  invalid_idx_mask = np.logical_and(predictions != 0, predictions != 1)
  # For any prediction != 0 or 1, set it to the opposite of what the target is
  predictions[invalid_idx_mask] = 1 - targets[invalid_idx_mask]
  return {"F1": f1_score(targets, predictions)}

def Converter(a):
    result_list=[]
    for i in a:
        if i in['True','true']:
            result_list.append(1)
        elif i in['False','false']:
            result_list.append(0)
        else:
            result_list.append(99)
            
    return result_list

def multirc_f1_over_all_answers(targets, predictions):
  """Special metric for MultiRC which computes F1 score over all examples.
  This is necessary because the targets/predictions for MultiRC are dicts and
  the f1_score_with_invalid expects a list of True/False labels, not dicts. As
  a result we just need to key in the "value" for each of the example dicts
  before feeding into f1_score_with_invalid.
  Args:
    targets: list of dicts, where each dict has a "value" key.
    predictions: list of dicts, where each dict has a "value" key.
  Returns:
    F1 score over values, where any prediction != 0 or 1 is counted as wrong.
  """
  return f1_score_with_invalid(
      Converter([t["original_target"] for t in targets]), Converter([p["original_target"] for p in predictions])
  )