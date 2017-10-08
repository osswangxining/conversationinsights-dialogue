# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import io
import json
import logging
import os
import random
import re
from collections import defaultdict
from collections import deque

from typing import Optional, List, Text

from conversationinsights.actions.action import ActionListen, ACTION_LISTEN_NAME
from conversationinsights.conversation import Dialogue
from conversationinsights.events import SetSlot, UserUtterance, ExecutedAction, Event, PauseConversation, Restart, \
    ResumeConversation, \
    RevertLastUserUtterance, SetTopic, ResetAllSlots, Reminder
from conversationinsights.interpreter import RegexInterpreter

logger = logging.getLogger(__name__)

# Checkpoint used to identify story starting blocks
STORY_START = "STORY_START"


class StoryParseError(Exception):
    """Raised if there is an error while parsing the story file."""

    def __init__(self, message):
        self.message = message


class StoryStep(object):
    def __init__(self, block_name=None, start_checkpoints=None, end_checkpoints=None, variation_idx=0, events=None):
        self.end_checkpoints = end_checkpoints if end_checkpoints else []
        self.start_checkpoints = start_checkpoints if start_checkpoints else []
        self.events = events if events else []
        self.block_name = block_name
        self.variation_idx = variation_idx

    def add_user_message(self, user_message):
        self.add_event(user_message)

    def add_event(self, event):
        # stories never contain the action listen events - they are implicit and added after a story is read and
        # converted to a dialogue
        if not isinstance(event, ExecutedAction) or event.action_name != ACTION_LISTEN_NAME:
            self.events.append(event)

    def num_actions(self):
        return sum(1 for e in self.events if isinstance(e, ExecutedAction))

    def as_story_string(self, flat=False):
        # if the result should be flattened, we will exclude the caption and any checkpoints.
        if flat:
            result = ""
        else:
            result = "\n## {}\n".format(self.block_name)
            for cp in self.start_checkpoints:
                if cp != STORY_START:
                    result += "> {}\n".format(cp)
        for s in self.events:
            if isinstance(s, UserUtterance):
                result += "* {}\n".format(s.as_story_string())
            elif isinstance(s, Event):
                result += "    - {}\n".format(s.as_story_string())
            else:
                raise Exception("Unexpected element in story step: " + s)

        if not flat:
            for cp in self.end_checkpoints:
                result += "> {}\n".format(cp)
        return result


class StoryBlock(object):
    def __init__(self, name):
        self.name = name
        self.turns = []
        self.current_turns = []
        self.start_checkpoints = []

    def add_checkpoint(self, name):
        # Depending on the state of the story part this is either a start or an end check point
        if not self.current_turns:
            self.start_checkpoints.append(name)
        else:
            for t in self.current_turns:
                t.end_checkpoints.append(name)

    def _prev_end_checkpoints(self):
        if not self.current_turns:
            return self.start_checkpoints
        elif self.current_turns[-1].end_checkpoints:
            return self.current_turns[-1].end_checkpoints
        else:
            return None

    def add_user_messages(self, messages):
        start_checkpoints = self._prev_end_checkpoints()
        if start_checkpoints is not None:
            # there is an end checkpoints in the prev turn, so we need to create a new one
            self.current_turns = self._creat_next_story_step()

        if len(messages) == 1:
            # If there is only one possible intent, we'll keep things simple
            for t in self.current_turns:
                t.add_user_message(UserUtterance(messages[0]))
        else:
            # If there are multiple different intents the user can use the express the same thing
            # we need to copy the blocks and create one copy for each possible message
            updated_turns = []
            for t in self.current_turns:
                for m in messages:
                    copied = copy.deepcopy(t)
                    copied.add_user_message(UserUtterance(m))
                    updated_turns.append(copied)
            self.current_turns = updated_turns

    def add_event(self, event):
        if not self.current_turns:
            self.current_turns = self._creat_next_story_step()
        for t in self.current_turns:
            t.add_event(event)

    def flush(self):
        if self.current_turns:
            self._add_current_stories_to_result()
            self.current_turns = []

    def _creat_next_story_step(self):
        self._add_current_stories_to_result()
        start_checkpoints = self._prev_end_checkpoints()
        current_turns = [StoryStep(block_name=self.name,
                                   start_checkpoints=start_checkpoints if start_checkpoints else [STORY_START])]
        return current_turns

    def _add_current_stories_to_result(self):
        for idx, turn in enumerate(self.current_turns):
            turn.variation_idx = idx
        self.turns.extend(self.current_turns)


class Story(object):
    def __init__(self, story_steps):
        self.story_steps = story_steps

    def used_actions(self):
        for step in self.story_steps:
            for s in step.events:
                if isinstance(s, ExecutedAction):
                    yield s.action_name

    def num_actions(self):
        c = 0
        for step in self.story_steps:
            c += step.num_actions()
        return c

    def as_dialogue(self, sender, domain, interpreter=RegexInterpreter()):
        events = []
        for step in self.story_steps:
            for e in step.events:
                if isinstance(e, UserUtterance):
                    parse_data = interpreter.parse(e.text)
                    updated_utterance = UserUtterance(e.text, parse_data["intent"], parse_data["entities"], parse_data)
                    events.append(ExecutedAction(ActionListen().id_str()))
                    events.append(updated_utterance)
                    events.extend(domain.slots_for_entities(parse_data["entities"]))
                elif isinstance(e, Event):
                    events.append(e)
                else:
                    raise Exception("Unexpected element in story step: " + e)

        events.append(ExecutedAction(ActionListen().id_str()))
        return Dialogue(sender, events)

    def as_story_string(self, flat=False):
        story_content = ""
        for step in self.story_steps:
            story_content += step.as_story_string(flat)

        if flat:
            return "## Generated Story {}\n{}".format(hash(story_content), story_content)
        else:
            return story_content

    def dump_to_file(self, file_name, flat=False):
        with io.open(file_name, "a") as f:
            f.write(self.as_story_string(flat))


class StoryFileReader(object):
    def __init__(self, lines, available_bot_actions=None, template_variables=None):
        self.story_parts = []
        self.current_story_block = None  # type: Optional[StoryBlock]
        self.available_bot_actions = available_bot_actions
        self.template_variables = template_variables if template_variables else {}
        self._process_lines(lines)

    @staticmethod
    def read_from_file(file_name, available_bot_actions=None, template_variables=None):
        try:
            with(io.open(file_name, "r")) as f:
                content = f.read()
            lines = content.splitlines()
            return StoryFileReader(lines, available_bot_actions, template_variables)
        except Exception as e:
            raise Exception("Failed to parse '{}'. {}".format(os.path.abspath(file_name), e))

    def _parse_event_line(self, line, parameter_default_value=""):
        m = re.search('^([^\[{]+)([\[{].+)?', line)
        if m is not None:
            event_name = m.group(1).strip()
            slots_str = m.group(2)
            parameters = {}
            if slots_str is not None and slots_str.strip():
                parsed_slots = json.loads(slots_str)
                if isinstance(parsed_slots, list):
                    for slot in parsed_slots:
                        parameters[slot] = parameter_default_value
                elif isinstance(parsed_slots, dict):
                    parameters = parsed_slots
                else:
                    raise Exception("Invalid slot string in line '{}'.".format(line))
            return event_name, parameters
        else:
            logger.debug("Failed to parse action line '{}'. ".format(line))
            return "", {}

    def _process_lines(self, lines):
        for idx, line in enumerate(lines):
            line_num = idx + 1
            try:
                line = self._replace_template_variables(self._clean_up_line(line))
                if line.strip() == "":
                    continue
                elif line.startswith("#"):
                    name = line[1:].strip("# ")
                    self.new_story_part(name)
                elif line.startswith(">"):
                    checkpoint_name = line[1:].strip()
                    logger.info("checkpoint_name: {}".format(checkpoint_name))
                    self.add_checkpoint(checkpoint_name)
                elif line.startswith("-"):
                    event_name, parameters = self._parse_event_line(line[1:])
                    self.add_event(event_name, parameters)
                elif line.startswith("*"):
                    user_messages = [el.strip() for el in line[1:].split(" OR ")]
                    self.add_user_messages(user_messages)
                else:
                    logger.warn("Skipping line {}. No valid command found. Line Content: '{}'".format(line_num, line))
            except Exception as e:
                msg = "Error in line {}: {}".format(line_num, e.message)
                logger.error(msg)
                raise Exception(msg)
        self._add_current_stories_to_result()

    def _replace_template_variables(self, line):
        def process_match(matchobj):
            varname = matchobj.group(1)
            if varname in self.template_variables:
                return self.template_variables[varname]
            else:
                raise ValueError("Unknown variable `{}` in template line '{}'".format(varname, line))

        template_rx = re.compile(r"`([^`]+)`")
        return template_rx.sub(process_match, line)

    def _clean_up_line(self, line):
        """Removes comments and trailing spaces"""
        return re.sub(r'<!--.*?-->', '', line).strip()

    def _add_current_stories_to_result(self):
        if self.current_story_block:
            self.current_story_block.flush()
            self.story_parts.append(self.current_story_block)

    def new_story_part(self, name):
        self._add_current_stories_to_result()
        self.current_story_block = StoryBlock(name)

    def add_checkpoint(self, checkpoint_name):
        # Ensure story part already has a name
        if not self.current_story_block:
            raise StoryParseError("Checkpoint '{}' at invalid location. Expected story start.".format(checkpoint_name))

        self.current_story_block.add_checkpoint(checkpoint_name)

    def add_user_messages(self, messages):
        if not self.current_story_block:
            raise StoryParseError("User message '{}' at invalid location. Expected story start.".format(messages))
        self.current_story_block.add_user_messages(messages)

    def add_event(self, event_name, parameters):
        if event_name == "pause":
            self.current_story_block.add_event(PauseConversation())
        elif event_name == "reset_slots":
            self.current_story_block.add_event(ResetAllSlots())
        elif event_name == "restart":
            self.current_story_block.add_event(Restart())
        elif event_name == "resume":
            self.current_story_block.add_event(ResumeConversation())
        elif event_name == "revert":
            self.current_story_block.add_event(RevertLastUserUtterance())
        elif event_name == "topic":
            topic = list(parameters.keys())[0] if parameters else ""
            self.current_story_block.add_event(SetTopic(topic))
        elif event_name == "reminder":
            logger.info("Reminders will be ignored during training, which should be ok")
            self.current_story_block.add_event(Reminder(parameters["action"],
                                                        parameters["date_time"],
                                                        parameters.get("id", None),
                                                        parameters.get("kill_on_user_msg", True),))
        elif event_name == "slot":
            slot_key = list(parameters.keys())[0] if parameters else None
            if slot_key:
                self.current_story_block.add_event(SetSlot(slot_key, parameters[slot_key]))
        else:
            if self.available_bot_actions is not None and event_name not in self.available_bot_actions:
                raise StoryParseError("Unknown event '{}' (neither an event nor an action).".format(event_name))
            self.current_story_block.add_event(ExecutedAction(event_name))

    def build_stories(self,
                      start_checkpoint=STORY_START,
                      remove_duplicates=True,
                      augmentation_factor=20,
                      max_history=1):
        # type: (Text, bool, bool, int) -> List[Story]
        """Uses the story parts read from a file to generate complete stories (e.g. resolving checkpoints)."""

        return StoryBuilder(self.story_parts).build_stories(
                start_checkpoint,
                remove_duplicates,
                augmentation_factor,
                max_history
        )


class StoryBuilder(object):
    def __init__(self, story_parts):
        self.story_parts = story_parts

    def build_stories(self,
                      start_checkpoint=STORY_START,
                      remove_duplicates=True,
                      augmentation_factor=20,
                      max_history=1):
        # type: (Text, bool, int, int) -> List[Story]
        """Given a set of story parts, generates all stories that are possible.

        The different story parts can end and start with checkpoints and this generator will
        match start and end checkpoints to connect complete stories. Afterwards, duplicate stories
        will be removed and the data is augmented (if augmentation is enabled)."""

        # keep track of the stories that are still missing their start checkpoint
        checkpoints = self._collect_checkpoints(self.story_parts)

        triggered_checkpoints = {start_checkpoint}
        resulting_stories = []          # type: List[Story]
        # start with the stories that do not have any prerequisites
        queue = [([], [cp]) for cp in checkpoints[start_checkpoint]]

        while queue:
            path, turns = queue.pop(0)

            if not turns[-1].end_checkpoints:
                # this is the end of a story
                prepared_turns = self._mark_first_action_as_unpredictable(turns)
                resulting_stories.append(Story(prepared_turns))
            else:
                for next_path, next_story in self._enqueue_triggered_story_blocks(path, turns,
                                                                                  checkpoints, triggered_checkpoints,
                                                                                  max_history):
                    queue.append((next_path, next_story))

        self._issue_unused_checkpoint_notification(triggered_checkpoints)

        if remove_duplicates:
            resulting_stories = self._remove_duplicates(resulting_stories)  # remove duplicates if required
        if augmentation_factor > 1:
            resulting_stories = self._augment_stories(resulting_stories, augmentation_factor, max_history)
        return resulting_stories

    def _mark_first_action_as_unpredictable(self, steps):
        """Marks actions that should not be used for a training of a predictive classifier.

        If a story starts with an action, we can not use that first action as a training example, as there is no
        history. There is one exception though, we do want to predict action listen. But because stories never
        contain action listen events (they are added when a story gets converted to a dialogue) we need to apply a
        small trick to avoid marking actions occurring after an action listen as unpredictable."""

        for j, step in enumerate(steps):
            for i, e in enumerate(step.events):
                if isinstance(e, UserUtterance):
                    # if there is a user utterance, that means before the user uttered something there has to be
                    # an action listen. therefore, any action that comes after this user utterance isn't the first
                    # action anymore and the tracker used for prediction is not empty anymore. Hence, it is fine
                    # to predict anything that occurs after a user utterance.
                    return steps
                if isinstance(e, ExecutedAction):
                    # we don't want to modify the original array - let's create a copy
                    cloned = steps[:]
                    cloned[j] = copy.deepcopy(step)
                    cloned[j].events[i].unpredictable = True
                    return cloned
        return steps

    def _has_cycle_in_path(self, checkpoints, current_checkpoint, story, max_history):
        # if we have encountered any checkpoint twice, then we are in a loop!
        try:
            first_occurence = checkpoints.index(current_checkpoint)
            num_actions_between_checkpoints = sum(s.num_actions() for s in story[first_occurence+1:])
            return checkpoints.count(current_checkpoint) > 2 and num_actions_between_checkpoints >= max_history
        except ValueError:
            return False

    def _enqueue_triggered_story_blocks(self, path, story, start_points,
                                        triggered_checkpoints, max_history):
        turn = story[-1]
        for checkpoint in turn.end_checkpoints:
            triggered_checkpoints.add(checkpoint)
            for p in start_points.get(checkpoint, []):
                if not self._has_cycle_in_path(path, checkpoint, story, max_history):
                    yield path[:] + [checkpoint], story[:] + [p]

            if checkpoint not in start_points:
                logger.warn("Unused checkpoint '{}' in block '{}'.".format(checkpoint, turn.block_name))

    def _issue_unused_checkpoint_notification(self, triggered_checkpoints):
        # Warns about unused components (having a start checkpoint that no one provided)
        for p in self.story_parts:
            for t in p.turns:
                for start_checkpoint in t.start_checkpoints:
                    if start_checkpoint not in triggered_checkpoints:
                        # After processing, there shouldn't be a story part left. This indicates a
                        # start checkpoint that doesn't exist
                        logger.warn("Unsatisfied start checkpoint '{}' in block '{}'".format(start_checkpoint, p.name))

    def _collect_checkpoints(self, story_parts):
        checkpoints = defaultdict(list)
        for p in story_parts:
            for turn in p.turns:
                for start in turn.start_checkpoints:
                    checkpoints[start].append(turn)
        return checkpoints

    def _remove_duplicates(self, stories):
        # type: (List[Story]) -> List[Story]
        """Removes duplicates from the list of passed stories."""

        deduplicated_stories = []
        seen_hashes = set()
        for s in stories:
            hashed = hash(s.as_story_string(flat=True))
            if hashed not in seen_hashes:
                seen_hashes.add(hashed)
                deduplicated_stories.append(s)
        if len(deduplicated_stories) != len(stories):
            logger.debug("Removed {} duplicates after story generation. {} remaining stories.".format(
                    len(stories) - len(deduplicated_stories), len(deduplicated_stories)))
        return deduplicated_stories

    def _augment_stories(self, stories, augmentation_factor, max_history):
        # type: (List[Story], int, int) -> List[Story]
        """Increase the number of stories by randomly concatenating them to one another (and clearing slots in between)

        Combines multiple stories by appending the stories. Stories are cleared between them."""

        random.seed(42)     # to make augmentation reproducible

        augmented_stories = []
        # collects tuples of (history_length, story_steps) of stories we need to augment
        # if the history_length reaches max_history the augmentation is stopped and the story is appended to the result
        to_be_augmented = deque()
        for s in stories:
            # last action of the current story already counts towards the history limit
            to_be_augmented.append((1, s.story_steps))
            augmented_stories.append(s)

        while len(to_be_augmented) > 0 and augmentation_factor > 0:
            added_actions, story_steps = to_be_augmented.pop()
            if added_actions >= max_history or len(story_steps) > max_history * 10:
                # story is already long enough, lets not extend it further but rather add it to the result
                augmented_stories.append(Story(story_steps))
            else:
                random.shuffle(stories)
                # we can't exhaustively explore all combinations, so we are going to randomly pick story sequences
                for s in stories[:augmentation_factor]:
                    story_copy = story_steps[:]
                    # It is debatable if the slots should be reset between the appended stories.
                    # The advantage of not doing it - is that the model gets more robust against variations in the slot
                    # values. Downside is, that it might pick up wrong flows.
                    # story_copy.append(StoryStep("slot-reset", events=[ResetAllSlots()]))
                    story_copy.extend(s.story_steps)
                    to_be_augmented.append((added_actions + s.num_actions(), story_copy))
        return augmented_stories
