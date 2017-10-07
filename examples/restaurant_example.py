# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from conversationinsights.actions.action import Action
from conversationinsights.policies.keras_policy import KerasPolicy

logger = logging.getLogger(__name__)


class ActionSearchRestaurants(Action):
    @classmethod
    def name(cls):
        return 'search_restaurants'

    @classmethod
    def run(cls, dispatcher, tracker, domain):
        dispatcher.utter_message("这是满足您要求的几个")
        return []


class ActionSuggest(Action):
    @classmethod
    def name(cls):
        return 'suggest'

    @classmethod
    def run(cls, dispatcher, tracker, domain):
        dispatcher.utter_message("前门全聚德烤鸭店")
        return []


class RestaurantPolicy(KerasPolicy):
    def _build_model(self, num_features, num_actions, max_history_len):
        """Build a keras model and return a compiled model.
        :param max_history_len: The maximum number of historical turns used to decide on next action"""
        from keras.layers import LSTM, Activation, Masking, Dense
        from keras.models import Sequential

        n_hidden = 32  # size of hidden layer in LSTM
        # Build Model
        model = Sequential()
        model.add(Masking(-1, batch_input_shape=(None, max_history_len, num_features)))
        model.add(LSTM(n_hidden, batch_input_shape=(None, max_history_len, num_features)))
        model.add(Dense(input_dim=n_hidden, output_dim=num_actions))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        logger.debug(model.summary())
        return model
