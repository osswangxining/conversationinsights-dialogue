from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from conversationinsights.domain import TemplateDomain
from conversationinsights.training_utils import create_stories_from_file
from conversationinsights.featurizers import BinaryFeaturizer
import numpy as np
import logging
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
def test_create_train_data_no_history(default_domain):
    logger.info("test_create_train_data_no_history")
    featurizer = BinaryFeaturizer()
    stories = create_stories_from_file("data/dsl_stories/stories_defaultdomain2.md",
                                       augmentation_factor=0)
    for story in stories:
        logger.info(story.as_story_string(True))
        logger.info(story.as_story_string())

    max_history = 1
    X, y = default_domain.training_data_from_stories(featurizer, stories, max_history, should_remove_duplicates=False)
    reference = np.array([
        [[0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 1, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 1, 0]],
        [[0, 1, 0, 0, 1, 0, 0, 0, 0]],
        [[0, 1, 0, 0, 0, 0, 1, 0, 0]],
        [[0, 0, 1, 0, 1, 0, 0, 0, 0]],
        [[0, 0, 1, 0, 0, 0, 0, 0, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 1, 0, 0, 0, 0]],
        [[1, 0, 0, 0, 0, 0, 0, 1, 0]],
        [[0, 1, 0, 0, 1, 0, 0, 0, 0]],
        [[0, 1, 0, 0, 0, 0, 1, 0, 0]]
    ])
    logger.info("reference.shape: {}".format(reference.shape))
    logger.info("y: {}".format(y))
    #DEBUG:conversationinsights.domain:Found 12 action examples: [0, 3, 0, 2, 0, 4, 0, 0, 3, 0, 2, 0]
    assert X.shape == reference.shape
    assert np.array_equal(X, reference)


def test_create_train_data_with_history(default_domain):
    featurizer = BinaryFeaturizer()
    stories = create_stories_from_file("data/dsl_stories/stories_defaultdomain2.md",
                                       augmentation_factor=0)
    max_history = 4
    X, y = default_domain.training_data_from_stories(featurizer, stories, max_history, should_remove_duplicates=False)
    reference = np.array([
        [[-1, -1, -1, -1, -1, -1, -1, -1, -1],
          [-1,-1,-1,-1,-1,-1,-1,-1, -1],
          [-1,-1,-1,-1,-1,-1,-1,-1, -1],
          [ 0,  0,  0,  0,  0,  0,  0,  0, 0]],

         [[-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 1, 0, 0, 0, 0]],

         [[-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 1, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 0, 0, 0, 1, 0]],

         [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 1, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 0, 0, 0, 1, 0],
          [ 0, 1, 0, 0, 1, 0, 0, 0, 0]],

         [[ 1, 0, 0, 0, 1, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 0, 0, 0, 1, 0],
          [ 0, 1, 0, 0, 1, 0, 0, 0, 0],
          [ 0, 1, 0, 0, 0, 0, 1, 0, 0]],

         [[ 1, 0, 0, 0, 0, 0, 0, 1, 0],
          [ 0, 1, 0, 0, 1, 0, 0, 0, 0],
          [ 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [ 0, 0, 1, 0, 1, 0, 0, 0, 0]],

         [[ 0, 1, 0, 0, 1, 0, 0, 0, 0],
          [ 0, 1, 0, 0, 0, 0, 1, 0, 0],
          [ 0, 0, 1, 0, 1, 0, 0, 0, 0],
          [ 0, 0, 1, 0, 0, 0, 0, 0, 1]],

         [[-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [ 0, 0, 0, 0, 0, 0, 0, 0, 0]],

         [[-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 1, 0, 0, 0, 0]],

         [[-1,-1,-1,-1,-1,-1,-1,-1,-1],
          [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 1, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 0, 0, 0, 1, 0]],

         [[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 1, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 0, 0, 0, 1, 0],
          [ 0, 1, 0, 0, 1, 0, 0, 0, 0]],

         [[ 1, 0, 0, 0, 1, 0, 0, 0, 0],
          [ 1, 0, 0, 0, 0, 0, 0, 1, 0],
          [ 0, 1, 0, 0, 1, 0, 0, 0, 0],
          [ 0, 1, 0, 0, 0, 0, 1, 0, 0]],
    ])
    assert X.shape == reference.shape
    assert np.array_equal(X, reference)


def test_domain_from_template():
    file = "examples/restaurant_domain.yml"
    domain = TemplateDomain.load(file)
    logger.info(domain.intents)
    assert len(domain.intents) == 6
    assert len(domain.actions) == 18
