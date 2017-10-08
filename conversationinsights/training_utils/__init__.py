from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from conversationinsights.training_utils.dsl import StoryFileReader, STORY_START


def create_stories_from_file(filename,
                             available_bot_actions=None,
                             start_checkpoint=STORY_START,
                             augmentation_factor=20,
                             max_history=1,
                             remove_duplicates=True):
    from conversationinsights.training_utils.dsl import StoryBuilder
    story = StoryFileReader.read_from_file(filename, available_bot_actions)
    return story.build_stories(
            start_checkpoint,
            remove_duplicates,
            augmentation_factor,
            max_history
    )
