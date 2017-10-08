from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging
import os

import numpy as np
from builtins import object
from typing import Text, List

from conversationinsights.domain import check_domain_sanity
from conversationinsights.util import create_dir_for_file

logger = logging.getLogger(__name__)


class PolicyTrainer(object):
    def __init__(self, ensemble, domain, featurizer):
        self.domain = domain
        self.ensemble = ensemble
        self.featurizer = featurizer

    def train(self, filename=None, max_history=3, augmentation_factor=20, max_training_samples=None, **kwargs):
        """Trains a policy on a domain using the training data contained in a file.

        :param augmentation_factor: how many stories should be created by randomly concatenating stories to one another
        :param filename: story file containing the training conversations
        :param max_history: number of past actions to consider for the prediction of the next action
        :param max_training_samples: specifies how many training samples to train on - `None` to use all examples
        :param kwargs: additional arguments passed to the underlying ML trainer (e.g. keras parameters)
        :return: trained policy
        """

        logger.debug("Policy trainer got kwargs: {}".format(kwargs))
        check_domain_sanity(self.domain)

        X, y = self._prepare_training_data(filename, max_history,
                                           augmentation_factor, max_training_samples)

        self.ensemble.train(X, y, self.domain, self.featurizer, **kwargs)

    def _prepare_training_data(self, filename, max_history, augmentation_factor,
                               max_training_samples=None,
                               should_remove_duplicates=True):
        """Reads the training data from file and prepares it for the training."""

        from conversationinsights.training_utils import create_stories_from_file

        if filename:
            stories = create_stories_from_file(filename,
                                               augmentation_factor=augmentation_factor,
                                               max_history=max_history, remove_duplicates=should_remove_duplicates)
            X, y = self.domain.training_data_from_stories(
                self.featurizer, stories, max_history, should_remove_duplicates)
            if max_training_samples is not None:
                X = X[:max_training_samples, :]
                y = y[:max_training_samples]
        else:
            X = np.zeros((0, self.domain.num_features))
            y = np.zeros(self.domain.num_actions)
        return X, y
