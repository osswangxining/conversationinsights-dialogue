from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from conversationinsights.policies.keras_policy import KerasPolicy
from conversationinsights.util import print_color, bcolors
from conversationinsights.domain import TemplateDomain
from conversationinsights.featurizers import BinaryFeaturizer
from conversationinsights.controller import Controller
from conversationinsights.interpreter import MyNLUHttpInterpreter
from conversationinsights.tracker_store import InMemoryTrackerStore

import logging
logger = logging.getLogger(__name__)

# paths
data_dir = "./examples/phonebook/data"

db_file = data_dir + "/phonebook.db"
fe_file = data_dir + "/total_word_feature_extractor.dat"
weights_file = data_dir + "/weights.h5"
arch_file = data_dir + "/arch.json"
classifier_file = data_dir + "/classifier.dat"
ner_file = data_dir + "/ner.dat"


# create an domain instance
# domain = PhonebookDomain(db_file)


# create the policy, interpreter, and controller instances
domain = TemplateDomain.load("examples/default_domain.yml")
#policy = Policy(weights_file,arch_file,default_domain)
featurizer = BinaryFeaturizer()
p = KerasPolicy()
policy = p.load("examples/babi/models/policy/current", featurizer, max_history=3)
interpreter = MyNLUHttpInterpreter("default","token", "http://127.0.0.1:5000")
controller = Controller(interpreter, policy, domain, InMemoryTrackerStore(domain))


#def test_hello():
#    tracker = controller.handle_message("")
#    response = controller.get_next_action(tracker, "hi")
#    response = controller.handle_message("hi")
#    logger.info("response:{}".format(response))
#    assert response == "utter_hello"
