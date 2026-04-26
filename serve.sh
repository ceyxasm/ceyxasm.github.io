#!/bin/bash
eval "$(rbenv init - bash)"
lsof -ti :4000 | xargs kill -9 2>/dev/null
bundle exec jekyll serve
