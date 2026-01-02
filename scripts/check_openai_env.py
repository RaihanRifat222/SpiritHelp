import os

def show(name):
    v = os.getenv(name)
    if not v:
        print(f"{name}: not set")
    else:
        # don't print full key, but show useful info for debugging
        sanitized = (v[:8] + "..." + v[-4:]) if len(v) > 20 else v
        print(f"{name}: set, len={len(v)}, preview={sanitized}, repr={repr(v)[:60]})")

names = ['OPENAI_API_KEY', 'OPENAI_KEY', 'OPENAI_API', 'OPENAI_API_BASE', 'OPENAI_API_TYPE', 'OPENAI_ORGANIZATION', 'OPENAI_ORG']
for n in names:
    show(n)

# Also show if dotenv file exists and its first line for context (don't reveal content)
try:
    with open('.env','r',encoding='utf-8') as f:
        first = f.readline().strip()
        print('\n.env first line (sanitized):', (first[:60] + '...') if len(first) > 60 else first)
except Exception as e:
    print('\n.env not found or not readable:', e)
