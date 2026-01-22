# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

from recbole.evaluator.register import metrics_dict
from recbole_cdr.evaluator.collector import DataStruct
from collections import OrderedDict

from logging import getLogger


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config['metrics']]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

        self.logger = getLogger()
        self.decimal_place = config['metric_decimal_place']

    def _get_metrics_with_topk(self, metric_val):
        """Helper function to get all metric names including @k variants"""
        metrics = {}
        for key, value in metric_val.items():
            metrics[key] = value
        return metrics

    def evaluate(self, dataobject: DataStruct):
        """Calculate all the metrics. It is called at the end of each epoch.

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: Contains overall, group1, and group2 evaluation results.
        """
        # Overall evaluation to get all metric names including @k variants
        result_dict = OrderedDict()
        sample_metrics = {}
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)
            sample_metrics.update(self._get_metrics_with_topk(metric_val))

        # Group 1 evaluation
        group1_result_dict = OrderedDict()
        try:
            for metric in self.metrics:
                metric_val = self.metric_class[metric].calculate_metric(dataobject, group="group1")
                group1_result_dict.update(metric_val)
        except (IndexError, ValueError) as e:
            self.logger.warning(f"Group 1 evaluation failed: {str(e)}. Using default values.")
            # Set default values for all metrics including @k variants
            for key in sample_metrics.keys():
                group1_result_dict[key] = -1.0
            self.logger.info("group1_result_dict: {}".format(group1_result_dict))

        # Group 2 evaluation
        group2_result_dict = OrderedDict()
        try:
            for metric in self.metrics:
                metric_val = self.metric_class[metric].calculate_metric(dataobject, group="group2")
                group2_result_dict.update(metric_val)
        except (IndexError, ValueError) as e:
            self.logger.warning(f"Group 2 evaluation failed: {str(e)}. Using default values.")
            # Set default values for all metrics including @k variants
            for key in sample_metrics.keys():
                group2_result_dict[key] = -1.0

        # UGF evaluation (unfairness gap between groups)
        ugf_result_dict = OrderedDict()
        for key in result_dict.keys():
            # If either group has default values (-1), set UGF to -1
            if group1_result_dict.get(key, -1.0) == -1.0 or group2_result_dict.get(key, -1.0) == -1.0:
                ugf = -1.0
            else:
                ugf = abs(group1_result_dict[key] - group2_result_dict[key])
            ugf_result_dict[key] = round(ugf, self.decimal_place)

        return {
            'overall': result_dict,
            'group1': group1_result_dict,
            'group2': group2_result_dict,
            'ugf': ugf_result_dict
        }