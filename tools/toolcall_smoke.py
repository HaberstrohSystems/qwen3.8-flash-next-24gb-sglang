"""One chat request with the mini-swe-agent bash tool: checks the reasoning/tool-call parsers on the local server."""
import json, urllib.request
tool = {"type": "function", "function": {"name": "bash", "description": "Run a bash command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}
body = {"model": "x", "messages": [{"role": "user", "content": "List the files in /tmp using the bash tool."}], "tools": [tool], "max_tokens": 2000, "temperature": 0.6}
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:30000/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
m = json.load(r)["choices"][0]["message"]
print("  reasoning:", (m.get("reasoning_content") or "")[:120].replace("\n", " "))
print("  content:", (m.get("content") or "")[:120].replace("\n", " "))
print("  tool_calls:", json.dumps(m.get("tool_calls"))[:200])
