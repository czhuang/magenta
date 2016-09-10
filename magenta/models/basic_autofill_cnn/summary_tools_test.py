"""Test summary_tools on edge cases."""

 
import numpy as np
import tensorflow as tf
import magenta.models.basic_autofill_cnn.summary_tools as summary_tools


class BasicAutofillCNNGenerateTest(tf.test.TestCase):

  def testAggregateMean(self):
 """Test that the aggregation leads to the correct mean."""
 values p.random.random(10)
 mean_aggregate ummary_tools.AggregateMean('test')
 for value in values:
   mean_aggregate.add(value, 1)
 self.assertAlmostEqual(np.mean(values), mean_aggregate.mean)

  def checkAggregateInOutMaskPredictionPerformanceStats(self, targets,
              predictions, mask):
 """Check that the on and off counts add up to size of pianoroll."""
 prediction_threshold .5
 perf ummary_tools.AggregateInOutMaskPredictionPerformanceStats(
  'testing', prediction_threshold)
 for n range(5):
   perf.add(predictions, targets, mask)

 for kind_str in ['', 'mask-']:
   self.assertEqual(
    perf.aggregates['%snoteons' ind_str].predicted_positive_counts +
    (perf.aggregates['%snoteoffs' %
       kind_str)].predicted_positive_counts),
    perf.aggregates['%snoteons' ind_str].instance_counts)

  def all_zeros(self):
 return np.zeros((32, 128))

  def all_ones(self):
 return np.ones((32, 128))

  def testAggregateInOutMaskPredictionPerformanceStats(self):
 """Test different combinations of targets and predictions."""
 target_cases self.all_zeros(), self.all_ones()]
 prediction_cases self.all_zeros(), self.all_ones(),
      self.all_ones() .5, self.all_ones() .49]
 for targets in target_cases:
   for predictions in prediction_cases:
  for mask in target_cases:
    self.checkAggregateInOutMaskPredictionPerformanceStats(targets,
                 predictions,
                 mask)


if __name__ == '__main__':
  tf.test.main()
