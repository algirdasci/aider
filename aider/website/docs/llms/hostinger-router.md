---
parent: Connecting to LLMs
nav_order: 500
---

# Hostinger Router

Aider can connect to models provided by [Hostinger Router](https://www.hostinger.com/ai-router),
an OpenAI-compatible multi-model API.
You'll need a Hostinger Router API key.

First, install aider:

{% include install.md %}

Then configure your API key:

Create a key in [hPanel](https://hpanel.hostinger.com/ai-router/api-keys), then:

```
export HOSTINGER_ROUTER_API_KEY=<key> # Mac/Linux
setx   HOSTINGER_ROUTER_API_KEY <key> # Windows, restart shell after setx
```

Start working with aider and Hostinger Router on your codebase:

```bash
# Change directory into your codebase
cd /to/your/project

aider --model hostinger_router/claude-sonnet-5

# List models available from Hostinger Router
aider --list-models hostinger_router/
```
