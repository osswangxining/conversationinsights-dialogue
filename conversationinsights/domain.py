from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import collections
import io
import json
import logging
import os

import numpy as np
from builtins import str
from conversationinsights.actions import Action
from conversationinsights.actions import DefaultTopic
from conversationinsights.actions.action import ActionListen, UtterAction, ActionRestart
from conversationinsights.conversation import Topic
from conversationinsights.events import ExecutedAction
from conversationinsights.featurizers import Featurizer
from conversationinsights.slots import Slot
from conversationinsights.trackers import DialogueStateTracker, SetSlot
from conversationinsights.util import lazyproperty, create_dir_for_file, class_from_module_path
from six import with_metaclass
from typing import Dict, Tuple
from typing import List
from typing import Optional
from typing import Text

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def check_domain_sanity(domain):
    """Makes sure the domain is properly configured and the chosen settings do make some sense.

    Checks the settings and checks if there are duplicate actions, intents, slots and entities."""

    def get_duplicates(my_items):
        """Returns a list of duplicate items in my_items."""
        return [item for item, count in collections.Counter(my_items).items() if count > 1]

    def get_exception_message(duplicates):
        """Returns a message given a list of pairs of (duplicate_actions [List], name [Text])."""
        msg = ""
        for d, name in duplicates:
            if d:
                if msg:
                    msg += "\n"
                msg += ("Duplicate {} in domain. ".format(name) +
                        "These {} occur more than once in the domain's `{}()` list: ".format(name, name) +
                        ", ".join(d))
        return msg

    duplicate_actions = get_duplicates([a.name() for a in domain.actions])
    duplicate_intents = get_duplicates([i for i in domain.intents])
    duplicate_slots = get_duplicates([s.name for s in domain.slots])
    duplicate_entities = get_duplicates([e for e in domain.entities])

    if duplicate_actions or duplicate_intents or duplicate_slots or duplicate_entities:
        raise Exception(get_exception_message([(duplicate_actions, "actions"),
                                               (duplicate_intents, "intents"),
                                               (duplicate_slots, "slots"),
                                               (duplicate_entities, "entitites")]))


class Domain(with_metaclass(abc.ABCMeta, object)):
    """The domain specifies the universe in which the bot's policy acts.

    A Domain subclass provides the actions the bot can take, the intents
    and entities it can recognise, and the topics it knows about."""

    DEFAULT_ACTIONS = [ActionListen(), ActionRestart()]

    # Type checker does not properly handle abstract properties, hence disabled for this function
    def __init__(self, topics=None, store_entities_as_slots=True, restart_intent="restart"):
        self.default_topic = DefaultTopic
        self.topics = topics if topics is not None else []
        self.store_entities_as_slots = store_entities_as_slots
        self.restart_intent = restart_intent

    @lazyproperty
    def num_actions(self):
        """Returns the number of available actions."""

        # noinspection PyTypeChecker
        return len(self.actions)

    @lazyproperty
    def action_map(self):
        # type: () -> Dict[Text, Tuple[int, Action]]
        """Provides a mapping from action names to indices and actions."""
        return {a.id_str(): (i, a) for i, a in enumerate(self.actions)}

    @lazyproperty
    def num_features(self):
        """Returns the number of used input features for the action prediction."""

        return len(self.input_features)

    def action_for_name(self, action_name):
        # type: (Text) -> Optional[Action]
        """Looks up which action corresponds to this action name."""

        return self.action_map.get(action_name)[1]

    def action_for_index(self, index):
        """A Policy will return an integer index corresponding to an action to be taken.

        This method resolves which action that corresponds to."""

        if len(self.actions) <= index or index < 0:
            raise Exception("Can not access action at index {}. Domain has {} actions.".format(
                    index, len(self.actions)))
        return self.actions[index]

    def index_for_action(self, action_name):
        # type: (Text) -> Optional[int]
        """Looks up which action index corresponds to this action name"""

        if action_name in self.action_map:
            return self.action_map.get(action_name)[0]
        else:
            raise Exception("Can not access action '{}', ".format(action_name) +
                            "as that name is not a registered action for this domain. Available actions are: \n" +
                            "\n".join(["\t - {}".format(a) for a in sorted(self.action_map)]))

    def training_data_from_stories(self, featurizer, stories, max_history, should_remove_duplicates):
        """Takes a list of stories created from the story DSL and creates a vector representation.

        The vector representation can be used for supervised learning.

        y is a 1D array of target labels
        X is a 3D array of shape (len(y),max_history,num_features)

        max_history specifies the number of previous steps to be included
        in the input. Fox max_history==1 you can flatten X to the canonical
        (num_points,num_features) shape.

        the point of the 3D shape is that it's what an RNN expects as input."""
        from conversationinsights.channels import UserMessage

        all_actions = []
        # create trackers at all of the intermediate states
        logger.debug('Generating features for trackers')
        state_features = []
        logger.debug("Creating training data from {} stories...".format(len(stories)))
        for story in stories:
            # create a dialogue for the story
            dialogue = story.as_dialogue(UserMessage.DEFAULT_SENDER, self)

            for event in dialogue.events:
                logger.debug("event: {}".format(event))
                if isinstance(event, ExecutedAction) and not event.unpredictable:
                    logger.debug("event.action_name: {}".format(event.action_name))
                    all_actions.append(self.index_for_action(event.action_name))

            logger.debug("slots: {}".format(self.slots))
            logger.debug("topics: {}".format(self.topics))
            logger.debug("default_topic: {}".format(self.default_topic.name))
            tracker = DialogueStateTracker(UserMessage.DEFAULT_SENDER,
                                           self.slots, self.topics, self.default_topic)
            tracker.update_with_dialogue(dialogue)
            # We need to drop the last one, as it is describing the state for the next action that is not part of
            # the story. (i.e. if the story wouldn't have ended at this point, this would be the tracker for the next
            # action).
            all_features = self.features_for_tracker_history(tracker)
            logger.debug("all_features: {}".format(all_features))

            for slice_end in range(1, len(all_features)):
                feature_vec = self.slice_feature_history(featurizer, all_features, slice_end, max_history)
                logger.debug("slice_end:{},max_history:{},feature_vec: {}".format(slice_end, max_history, feature_vec))
                state_features.append(feature_vec)

        assert len(state_features) == len(all_actions), "Got {} trackers and {} actions".format(len(state_features),
                                                                                                len(all_actions))
        logger.debug("Found state_features: {}".format(state_features))
        X = np.array(state_features)
        logger.debug("Found {} action examples: {}".format(len(all_actions), all_actions))
        y = np.array(all_actions)  # target labels are easy
        logger.debug("Found action examples: {}".format(y))

        if should_remove_duplicates:
            X_unique, y_unique = self._deduplicate_training_data(X, y)
            logger.debug("Deduplicated to {} unique action examples.".format(y_unique.shape[0]))
            return X_unique, y_unique
        else:
            return X, y

    def _deduplicate_training_data(self, X, y):
        """Makes sure every training example in X occurs only once with the same label."""
        # we need to concat X and y to make sure that we do NOT throw out contradicting examples
        # (same featurization but different labels). appends y to X so it appears to be just another feature
        casted_y = np.broadcast_to(np.reshape(y, (y.shape[0], 1, 1)), (y.shape[0], X.shape[1], 1))
        concatenated = np.concatenate((X, casted_y), axis=2)
        t_data = np.unique(concatenated, axis=0)
        X_unique = t_data[:, :, :-1]
        y_unique = np.array(t_data[:, 0, -1], dtype=casted_y.dtype)
        return X_unique, y_unique

    def slice_feature_history(self, featurizer, tracker_history_features, slice_end, slice_length):
        # type: (Featurizer, List[Dict[Text, float]], int, int) -> np.ndarray
        """Given an array of features for the history of a tracker a slice of the passed length will be extracted.

        If the slice is at the array borders, padding will be added to ensure the slice length."""

        pad_len = max(0, slice_length - slice_end)
        slice_start = max(0, slice_end - slice_length)
        logger.debug("slice_length:{},pad_len:{},slice_start:{},slice_end:{}".format(slice_length,pad_len,slice_start,slice_end))
        state_features = pad_len * [None] + tracker_history_features[slice_start:slice_end]
        logger.debug("state_features:{}".format(state_features))
        encoded_features = [featurizer.encode_features(f, self.input_feature_map) for f in state_features]
        logger.debug("encoded_features:{}".format(encoded_features))
        return np.vstack(encoded_features)

    def features_for_tracker_history(self, tracker):
        """Creates an array with the features for each state of the trackers history."""
        return [self.get_active_features(tr) for tr in tracker.generate_all_prior_states()]

    def feature_vector_for_tracker(self, featurizer, tracker, max_history):
        """Creates a 2D array of shape (max_history,num_features)

        max_history specifies the number of previous steps to be included
        in the input. Each row in the array corresponds to the binarised
        features of each state. Result is padded with default values if
        there are fewer than `max_history` states present."""

        all_features = self.features_for_tracker_history(tracker)
        return self.slice_feature_history(featurizer, all_features, len(all_features), max_history)

    def random_template_for(self, utter_action):
        if utter_action in self.templates:
            return np.random.choice(self.templates[utter_action])
        else:
            return None

    # noinspection PyTypeChecker
    @lazyproperty
    def slot_features(self):
        # type: () -> List[Text]
        """Returns all available slot feature strings."""

        return ["slot_{}_{}".format(s.name, i) for s in self.slots for i in range(0, s.feature_dimensionality())]

    # noinspection PyTypeChecker
    @lazyproperty
    def prev_action_features(self):
        # type: () -> List[Text]
        """Returns all available previous action feature strings."""

        return ["prev_{0}".format(a.id_str()) for a in self.actions]

    # noinspection PyTypeChecker
    @lazyproperty
    def intent_features(self):
        # type: () -> List[Text]
        """Returns all available previous action feature strings."""

        return ["intent_{0}".format(i) for i in self.intents]

    # noinspection PyTypeChecker
    @lazyproperty
    def entity_features(self):
        # type: () -> List[Text]
        """Returns all available previous action feature strings."""

        return ["entity_{0}".format(e) for e in self.entities]

    def index_of_feature(self, feature_name):
        # type: (Text) -> Optional[int]
        """Provides the index of a feature."""

        return self.input_feature_map.get(feature_name)

    @lazyproperty
    def input_feature_map(self):
        # type: () -> Dict[Text, int]
        """Provides a mapping from feature names to indices."""
        return {f: i for i, f in enumerate(self.input_features)}

    @lazyproperty
    def input_features(self):
        # type: () -> List[Text]
        """Returns all available features."""

        return self.intent_features + self.entity_features + self.slot_features + self.prev_action_features

    def get_active_features(self, tracker):
        # type: (DialogueStateTracker) -> Dict[Text, float]
        """Return a bag of active features from the tracker state"""
        logger.debug("tracker.export_stories: {}".format(tracker.export_stories()))
        feature_dict = self.get_parsing_features(tracker)
        feature_dict.update(self.get_prev_action_features(tracker))
        logger.debug("updated feature_dict:{}".format(feature_dict))
        return feature_dict

    def get_prev_action_features(self, tracker):
        # type: (DialogueStateTracker) -> Dict[Text, float]
        """Turns the previous taken action into a feature name."""
        logger.debug("tracker.latest_action_id_str:{}".format(tracker.latest_action_id_str))
        latest_action = tracker.latest_action_id_str
        if latest_action:
            if "prev_{}".format(latest_action) in self.input_feature_map:
                return {"prev_{}".format(latest_action): 1}
            else:
                raise Exception("Failed to use action '{}' in history. ".format(latest_action) +
                                "Please make sure all actions are listed in the domains action list.")
        else:
            return {}

    def get_parsing_features(self, tracker):
        # type: (DialogueStateTracker) -> Dict[Text, float]

        feature_dict = {}

        logger.debug("tracker.latest_message: {}".format(tracker.latest_message))
        # Set all found entities with the feature value 1.0
        for entity in tracker.latest_message.entities:
            key = "entity_{0}".format(entity["entity"])
            feature_dict[key] = 1.

        # Set all set slots with the featurization of the stored value
        for key, slot in tracker.slots.items():
            if slot is not None:
                for i, slot_value in enumerate(slot.as_feature()):
                    feature_dict["slot_{}_{}".format(key, i)] = slot_value

        latest_msg = tracker.latest_message
        logger.debug("latest_msg:{}".format(latest_msg))
        if "intent_ranking" in latest_msg.parse_data:
            for intent in latest_msg.parse_data["intent_ranking"]:
                if intent.get("name"):
                    feature_dict["intent_{}".format(intent["name"])] = intent["confidence"]

        elif latest_msg.intent.get("name"):
            feature_dict["intent_{}".format(latest_msg.intent["name"])] = latest_msg.intent.get("confidence", 1.0)

        logger.debug("feature_dict:{}".format(feature_dict))
        return feature_dict

    def slots_for_entities(self, entities):
        events = []
        if self.store_entities_as_slots:
            for entity in entities:
                for s in self.slots:
                    if entity['entity'] == s.name:
                        events.append(SetSlot(entity['entity'], entity['value']))
        return events

    def persist(self, file_name):
        raise NotImplementedError

    @classmethod
    def load(cls, file_name):
        raise NotImplementedError

    def persist_specification(self, model_path):
        # type: (Text, List[Text]) -> None
        """Persists the domain specification to storage."""

        domain_spec_path = os.path.join(model_path, 'domain.json')
        create_dir_for_file(domain_spec_path)
        metadata = {
            "features": self.input_features
        }
        with io.open(domain_spec_path, 'w') as f:
            f.write(str(json.dumps(metadata, indent=2)))

    @classmethod
    def load_specification(cls, path):
        matadata_path = os.path.join(path, 'domain.json')
        with io.open(matadata_path) as f:
            specification = json.loads(f.read())
        return specification

    def compare_with_specification(self, path):
        # type: (Text) -> bool
        """Compares the domain specifications of the current and the loaded ones.

        Throws exception if the loaded domain specification is different to the current domain are different."""

        loaded_domain_spec = self.load_specification(path)
        if loaded_domain_spec["features"] != self.input_features:
            diff = (set(loaded_domain_spec["features"]) - set(self.input_features)).union(
                set(self.input_features) - set(loaded_domain_spec["features"]))
            raise Exception("Domain specification has changed. You MUST retrain the policy. " +
                            "Detected mismatch in domain specification. " +
                            "The following has been removed or added: {}".format(", ".join(diff)))
        else:
            return True

    # Abstract Methods : These have to be implemented in any domain subclass

    @abc.abstractproperty
    def slots(self):
        # type: () -> List[Slot]
        """Domain subclass must provide a list of slots"""
        pass

    @abc.abstractproperty
    def entities(self):
        # type: () -> List[Text]
        raise NotImplementedError("domain subclass must provide a list of entities")

    @abc.abstractproperty
    def intents(self):
        # type: () -> List[Text]
        raise NotImplementedError("domain subclass must provide a list of intents")

    @abc.abstractproperty
    def actions(self):
        # type: () -> List[Action]
        raise NotImplementedError("domain subclass must provide a list of possible actions")

    @abc.abstractproperty
    def templates(self):
        # type: () -> List[Text]
        raise NotImplementedError("domain subclass must provide a dictionary of response templates")


class TemplateDomain(Domain):
    @classmethod
    def load(cls, file_name):
        import yaml
        import io
        if os.path.isfile(file_name):
            with io.open(file_name, encoding="utf-8") as f:
                data = yaml.load(f.read())
                templates = data.get("templates", [])
                topics = [Topic(name) for name in data.get("topics", [])]
                actions = TemplateDomain.collect_actions(templates, data.get("actions", []))
                slots = TemplateDomain.collect_slots(data.get("slots", {}))
                additional_arguments = data.get("config", {})
                return TemplateDomain(
                  data.get("intents", []),
                  data.get("entities", []),
                  slots,
                  templates,
                  actions,
                  topics,
                  **additional_arguments
                )
        else:
            raise Exception("Failed to load domain specification from '{}'. File not found!".format(
                    os.path.abspath(file_name)))

    @staticmethod
    def collect_actions(templates, action_names):
        actions = Domain.DEFAULT_ACTIONS[:]
        for name in action_names:
            if "utter_" + name in templates:
                actions.append(UtterAction(name))
            else:
                try:
                    cls = class_from_module_path(name)
                    actions.append(cls())
                except ImportError as e:
                    raise ValueError("Action '{}' doesn't correspond to a template or an action class. ".format(name) +
                                     "Error: {}".format(e))
                except AttributeError as e:
                    raise ValueError("Action '{}' doesn't correspond to a template or an action class. ".format(name) +
                                     "Found the module, but the module doesn't contain a class with this name. " +
                                     "Error: {}".format(e))

        return actions

    @staticmethod
    def collect_slots(slot_dict):
        # it is super important to sort the slots here!!! otherwise feature ordering is not consistent
        slots = []
        for slot_name in sorted(slot_dict):
            slot_class = Slot.resolve_by_type(slot_dict[slot_name].get("type"))
            if "type" in slot_dict[slot_name]:
                del slot_dict[slot_name]["type"]
            slot = slot_class(slot_name, **slot_dict[slot_name])
            slots.append(slot)
        return slots

    def __init__(self, intents, entities, slots, templates, actions, topics, **kwargs):
        self._intents = intents
        self._entities = entities
        self._slots = slots
        self._templates = templates
        self._actions = actions
        super(TemplateDomain, self).__init__(topics, **kwargs)

    def _action_references(self):
        action_references = []
        for action in self.actions[len(Domain.DEFAULT_ACTIONS):]:   # assumes default actions are present!!!
            if isinstance(action, UtterAction):
                action_references.append(action.name())
            else:
                action_references.append(action.__module__ + "." + action.__class__.__name__)
        return action_references

    def _slot_definitions(self):
        slots = {}
        for slot in self.slots:
            d = slot.additional_persistence_info()
            d["type"] = slot.type_name
            slots[slot.name] = d
        return slots

    def persist(self, file_name):
        import yaml

        additional_config = {"store_entities_as_slots": self.store_entities_as_slots}
        topic_names = [t.name for t in self.topics]

        domain_data = {
            "config": additional_config,
            "intents": self.intents,
            "entities": self.entities,
            "slots": self._slot_definitions(),
            "templates": self.templates,
            "topics": topic_names,
            "actions": self._action_references()
        }

        with io.open(file_name, 'w', encoding="utf-8") as yaml_file:
            yaml.safe_dump(domain_data, yaml_file, default_flow_style=False, allow_unicode=True)

    @lazyproperty
    def templates(self):
        return self._templates

    @lazyproperty
    def slots(self):
        return self._slots

    @lazyproperty
    def intents(self):
        return self._intents

    @lazyproperty
    def entities(self):
        return self._entities

    @lazyproperty
    def actions(self):
        return self._actions
