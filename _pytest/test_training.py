from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from conversationinsights.training_utils import create_stories_from_file
from conversationinsights.training_utils.visualization import visualize_stories
from conversationinsights.training_utils.visualization import _persist_graph

def test_story_visualization():
    stories = create_stories_from_file("data/dsl_stories/stories.md")
    generated_graph = visualize_stories(stories)
    assert len(generated_graph.nodes()) == 23
