"""Classes for summarizing the accuracy of predictions, mostly binary."""

import numpy as np


class AggregateSpecificationError(Exception):
  """Exception for when aggregate attributes are misspecified."""
  pass


class AggregateMean(object):
  """Aggregates values for mean."""

  def __init__(self, name):
    self.name = name
    self.value = 0.
    self.total_counts = 0

  def add(self, value, counts):
    """Add an amount to the total and also increment the counts."""
    self.value += value
    self.total_counts += counts

  @property
  def mean(self):
    """Return the mean."""
    return self.value / self.total_counts


class AggregatePredictionPerformanceStats(object):
  """Aggregate binary-only predictions to compute precision-recall scores.

  Args:
      name: String name of the target prediction being scored, such as masked
          noteons, unmasked noteoffs.
      class_label: The class being scored, either 0 or 1.
      on_prediction_threshold: The threshold for predicting a 1.

    Raises:
      AggregateSpecificationError: If class_label is not binary.
  """

  def __init__(self, name, class_label, on_prediction_threshold):
    if class_label not in [0, 1]:
      raise AggregateSpecificationError('Class label must be binary.')
    self.name = name
    self.class_label = class_label
    self.prediction_threshold = on_prediction_threshold
    if not class_label:
      self.prediction_threshold = 1 - self.prediction_threshold

    self.target_counts = 0
    self.predicted_positive_counts = 0
    self.true_positive_counts = 0
    self.instance_counts = 0

  def add(self, predictions, targets):
    """Compute and add prediction accuracy-related count statistics.

    Args:
      predictions: A matrix of predicted probabilities of the binary class.
      targets: A matrix of the desired outcomes, 0 or 1s.
    """
    if not self.class_label:
      predictions = 1 - predictions
      targets = 1 - targets

    self.target_counts += np.sum(targets)
    # Use equal or larger than for the first comparison to ensure noteon and
    # noteoff statistics are mutually exclusive.
    if self.class_label:
      predicted_positive_boolean_inds = (
          predictions >= self.prediction_threshold)
    else:
      predicted_positive_boolean_inds = predictions > self.prediction_threshold
    self.predicted_positive_counts += np.sum(predicted_positive_boolean_inds)
    self.true_positive_counts += (
        np.sum(targets[predicted_positive_boolean_inds]))
    self.instance_counts += np.prod(targets.shape)

  def get_precision_recall_f1score(self):
    """Compute the precision, recall and F1-score of the aggregated counts."""
    if self.target_counts == 0:
      recall = 1.0
    else:
      recall = self.true_positive_counts / float(self.target_counts)
    if self.predicted_positive_counts == 0:
      precision = 0.0
    else:
      precision = self.true_positive_counts / (
          float(self.predicted_positive_counts))
    return {'recall': recall,
            'precision': precision,
            'f1score': 2 * recall * precision / (recall + precision)}


class AggregateInOutMaskPredictionPerformanceStats(object):
  """Wrapper for aggregateing overall and in mask only performance stats.

  Args:
      experiment_type: String of experiment type, such as 'training', or
          'validation'.
      on_prediction_threshold: The threshold for predicting a 1.
  """

  def __init__(self, experiment_type, on_prediction_threshold):
    self.experiment_type = experiment_type

    # 'noteons' and 'noteoffs' are for context-only, aka outside of mask.
    self.kinds = ['noteons', 'noteoffs', 'mask-noteons', 'mask-noteoffs']
    self.aggregates = {}
    for kind in self.kinds:
      class_label = 'off' not in kind
      self.aggregates[kind] = AggregatePredictionPerformanceStats(
          kind, class_label, on_prediction_threshold)

    self.accuracy = AggregateMean('accuracy')
    self.mask_accuracy = AggregateMean('mask-accuracy')

  def add(self, predictions, targets, masks):
    """Add accuracy-related counts for in and out of mask predictions."""
    # Aggregate out of mask statistics.
    outside_masks = 1 - masks
    outside_mask_inds = outside_masks > 0.0
    outside_predictions = predictions[outside_mask_inds]
    outside_targets = targets[outside_mask_inds]

    self.aggregates['noteons'].add(outside_predictions, outside_targets)
    self.aggregates['noteoffs'].add(outside_predictions, outside_targets)

    # Aggregate in mask statistics.
    mask_inds = masks > 0.0
    mask_predictions = predictions[mask_inds]
    mask_targets = targets[mask_inds]

    self.aggregates['mask-noteons'].add(mask_predictions, mask_targets)
    self.aggregates['mask-noteoffs'].add(mask_predictions, mask_targets)

    self.accuracy.add(*self.get_accuracy(''))
    self.mask_accuracy.add(*self.get_accuracy('mask-'))

  def get_accuracy(self, prefix=''):
    """Compute overall accuracy, including both note ons and note offs.

    Args:
      prefix: The prefix to which set of targets being summarized.  The two
        available sets currently are the masked ('mask-') and all ('').

    Returns:
      correct_counts: The number of correctly predicted pianoroll cells.
      instance_counts: The number of total pianoroll cells.

    Raises:
      NameError: If specified prefix not one of that has been tracked.
    """
    # TODO(annahuang): Add function to allow iterating over prefixes.
    if prefix not in ['', 'mask-']:
      raise NameError('Prefix not found.')
    correct_counts = self.aggregates[
        prefix + 'noteons'].true_positive_counts + (
            self.aggregates[prefix + 'noteoffs'].true_positive_counts)
    instance_counts = self.aggregates[prefix + 'noteoffs'].instance_counts
    return correct_counts, instance_counts

  def get_aggregates_stats(self):
    """Return all statistics, such as overall accuracy, precision recall, F1."""
    stats = {}
    for aggregate_name, aggregate in self.aggregates.iteritems():
      scores = aggregate.get_precision_recall_f1score()
      for key, score in scores.iteritems():
        stats['%s-%s_%s' % (aggregate_name, self.experiment_type, key)] = score
    stats['accuracy_%s' % self.experiment_type] = self.accuracy.mean
    stats['accuracy_mask_%s' % self.experiment_type] = self.mask_accuracy.mean
    return stats
