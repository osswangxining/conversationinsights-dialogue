from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io

from conversationinsights.events import SetSlot, ExecutedAction

from conversationinsights.training_utils import create_stories_from_file
import logging
logger = logging.getLogger(__name__)


def test_can_read_test_story():
    stories = create_stories_from_file("data/dsl_stories/stories.md", augmentation_factor=0)
    assert len(stories) == 7
    logger.info("stories: {}".format(len(stories)))

    story = stories[2]  # this should be the story simple_story_with_only_end -> show_it_all
    assert len(story.story_steps) == 3
    assert story.story_steps[0].block_name == 'simple_story_with_only_end'
    assert story.story_steps[1].block_name == 'show_it_all'
    assert story.story_steps[2].block_name == 'show_it_all'

    assert len(story.story_steps[0].events) == 4
    assert story.story_steps[0].events[1] == ExecutedAction("do_something_with_hello")
    assert story.story_steps[0].events[2] == SetSlot("name", "peter")
    assert story.story_steps[0].events[3] == SetSlot("nice_person", "")

    for event in story.story_steps[1].events:
        logger.info(event)


def test_persist_and_read_test_story(tmpdir):
    stories = create_stories_from_file("data/dsl_stories/stories.md", augmentation_factor=0)
    out_path = tmpdir.join("persisted_story.md")
    logger.info("out_path:{}".format(out_path))
    with io.open(out_path.strpath, "w") as f:
        for story in stories:
            f.write(story.as_story_string(flat=False))

        recovered_stories = create_stories_from_file(out_path.strpath)
    for s, r in zip(stories, recovered_stories):
        assert s.as_story_string() == r.as_story_string()


def test_story_augmentation():
    stories = create_stories_from_file("data/dsl_stories/stories.md", augmentation_factor=20)
    assert len(stories) == 14

    i = 0
    for story in stories:
        logger.info("story[{}]".format(i))
        logger.info(story.as_story_string())
        i = i + 1

    story = stories[11]  # this should be one of the augmented stories with 3 story steps
    assert len(story.story_steps) == 3
    assert story.story_steps[0].block_name == 'simple_story_with_only_end'
    assert story.story_steps[1].block_name == 'show_it_all'
    assert story.story_steps[2].block_name == 'show_it_all'

    assert len(story.story_steps[0].events) == 4
    assert story.story_steps[0].events[1] == ExecutedAction("do_something_with_hello")
    assert story.story_steps[0].events[2] == SetSlot("name", "peter")
    assert story.story_steps[0].events[3] == SetSlot("nice_person", "")
