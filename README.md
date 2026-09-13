# Update with litellm for more agnostic LLM frontier models
consider this for later
```py
If you'd rather uv manage the whole project (auto-tracks deps in pyproject.toml instead of requirements.txt, handles the venv for you on every run):

bash
uv init --no-readme .
uv add litellm
uv run test_agent.py
uv run main.py
```


# LIKELY DATED!
---
title: Template Final Assignment
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference