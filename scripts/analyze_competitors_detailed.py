import urllib.request
import json

def get_file(repo, path):
    url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Error: {e}"

# Check Sayan's cleaning and modeling
print("=== Sayan clean_train.py ===")
print(get_file('sayan23bcy6/MKS_Hackathon', 'project/pipeline/clean_train.py')[:1500])

print("\n=== Sayan unconstrained.py ===")
print(get_file('sayan23bcy6/MKS_Hackathon', 'project/pipeline/unconstrained.py')[:1500])

print("\n=== Sayan models ===")
print(get_file('sayan23bcy6/MKS_Hackathon', 'project/pipeline/models/__init__.py')[:1500])
