from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from conversationinsights.featurizers import BinaryFeaturizer
from conversationinsights.policies.ensemble import PolicyEnsemble


def test_domain_spec_dm():
    model_path = 'examples/babi/models/policy/test'
    policy = PolicyEnsemble.load(model_path, BinaryFeaturizer())
    policy.persist('examples/babi/models/policy/test2')
