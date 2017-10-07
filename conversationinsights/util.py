from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from collections import deque

import errno
import six
from typing import Text

from conversationinsights.conversation import Topic
from builtins import input, range, str


def class_from_module_path(module_path):
    """Given the module path of a class and its name, tries to retrieve that class.

    The loaded class can be used to instanciate new objects. """
    import importlib

    # load the module, will raise ImportError if module cannot be loaded
    if "." in module_path:
        module_name, _, class_name = module_path.rpartition('.')
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        return getattr(m, class_name)
    else:
        return globals()[module_path]


def all_subclasses(cls):
    """Returns all known (imported) subclasses of a class."""

    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


def is_int(value):
    """Checks if a value is an integer.

    The type of the value is not important, it might be an int a string or a float."""

    try:
        return value == int(value)
    except Exception:
        return False


def lazyproperty(fn):
    """Allows to avoid recomputing a property over and over. Instead the result gets stored in a local var.

    Computation of the property will happen once, on the first call of the property. All succeeding calls will use
    the value stored in the private property."""

    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazyprop


def create_dir_for_file(file_path):
    # type: (Text) -> None
    """Creates any missing parent directories of this files path."""

    try:
        os.makedirs(os.path.dirname(file_path))
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


def one_hot(hot_idx, length, dtype=None):
    import numpy
    if hot_idx >= length:
        raise Exception("Can't create one hot. Index '{}' is out of range (length '{}')".format(hot_idx, length))
    r = numpy.zeros(length, dtype)
    r[hot_idx] = 1
    return r


def str_range_list(start, end):
    return [str(e) for e in range(start, end)]


def request_input(valid_values, prompt=None, max_suggested=3):
    def wrong_input_message():
        print("Invalid answer, only {}{} allowed\n".format(
                ", ".join(valid_values[:max_suggested]), ",..." if len(valid_values) > max_suggested else ""))

    while True:
        try:
            input_value = input(prompt) if prompt else input()
            if input_value not in valid_values:
                wrong_input_message()
                continue
        except ValueError:
            wrong_input_message()
            continue
        return input_value


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def wrap_with_color(text, color):
    return color + text + bcolors.ENDC


def print_color(text, color):
    print(wrap_with_color(text, color))


class TopicStack(object):
    def __init__(self, topics, iterable, default):
        self.topics = topics
        self.iterable = iterable
        self.topic_names = [t.name for t in topics]
        self.default = default
        self.dq = deque(iterable, len(topics))

    @property
    def top(self):
        if len(self.dq) < 1:
            return self.default
        return self.dq[-1]

    def __iter__(self):
        return self.dq.__iter__()

    def next(self):
        return self.dq.next()

    def __len__(self):
        return len(self.dq)

    def push(self, x):
        if isinstance(x, six.string_types):
            if x not in self.topic_names:
                raise ValueError("Unknown topic name: '{}', known topics in this domain are: {}".format(
                    x, self.topic_names))
            else:
                x = self.topics[self.topic_names.index(x)]

        elif not isinstance(x, Topic) or x not in self.topics:
            raise ValueError("Instance of type '{}' can not be used on the topic stact, not a valid topic!".format(
                type(x).__name__))

        while self.dq.count(x) > 0:
            self.dq.remove(x)
        self.dq.append(x)

    def pop(self):
        if len(self.dq) < 1:
            return None
        return self.dq.pop()
