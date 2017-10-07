from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import re

import requests
from builtins import str

logger = logging.getLogger(__name__)


class NaturalLanguageInterpreter(object):
    def parse(self, text):
        raise NotImplementedError("Interpreter needs to be able to parse messages into structured output.")

    @staticmethod
    def create(obj):
        if isinstance(obj, NaturalLanguageInterpreter):
            return obj
        return None


class RegexInterpreter(NaturalLanguageInterpreter):
    @staticmethod
    def extract_intent_and_entities(user_input):
        m = re.search('^_([^\[]+)(\[(.+)\])?', user_input)
        if m is not None:
            intent = m.group(1).lower()
            offset = m.start(3)
            entities_str = m.group(3)
            entities = []
            if entities_str is not None:
                for entity_str in entities_str.split(','):
                    for match in re.finditer('\s*(.+)\s*=\s*(.+)\s*', entity_str):
                        start = match.start(2) + offset
                        end = match.end(0) + offset
                        entity = {
                            "entity": match.group(1),
                            "start": start,
                            "end": end,
                            "value": match.group(2)}
                        entities.append(entity)

            return intent, entities
        else:
            return None, []

    def parse(self, text):
        intent, entities = self.extract_intent_and_entities(text)
        return {
            'text': text,
            'intent': {
                'name': intent,
                'confidence': 1.0,
            },
            'intent_ranking': [{
                'name': intent,
                'confidence': 1.0,
            }],
            'entities': entities,
        }


class MyNLUHttpInterpreter(NaturalLanguageInterpreter):
    def __init__(self, model_name, token, server):
        self.model_name = model_name
        self.token = token
        self.server = server

    def parse(self, text):
        """Parses a text message. Returns a default value if the parsing of the text failed."""

        default_return = {"intent": {"name": "", "confidence": 0.0}, "entities": [], "text": ""}
        result = self._nlu_http_parse(text)

        return result if result is not None else default_return

    def _nlu_http_parse(self, text):
        """Send a text message to a running NLU http server. Returns `None` on failure."""
        if not self.server:
            logger.error("Failed to parse text '{}' using NLU over http. No NLU server specified!".format(
                    text))
            return None

        params = {
            "token": self.token,
            "model": self.model_name,
            "q": text
        }
        url = "{}/parse".format(self.server)
        try:
            result = requests.get(url, params=params)
            if result.status_code == 200:
                return result.json()
            else:
                logger.error("Failed to parse text '{}' using NLU over http. Error: {}".format(text, result.text))
                return None
        except Exception as e:
            logger.error("Failed to parse text '{}' using NLU over http. Error: {}".format(text, e))
            return None
