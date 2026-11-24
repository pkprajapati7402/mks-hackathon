import urllib.request
import json
import os

def list_files(repo, path=''):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return [item['name'] for item in data]
    except Exception as e:
        return [f"Error: {e}"]

def get_file(repo, path):
    url = f"https://raw.githubusercontent.com/{repo}/main/{path}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    print("=== Sayan project/pipeline ===")
    sayan_pipeline = list_files('sayan23bcy6/MKS_Hackathon', 'project/pipeline')
    print(sayan_pipeline)

    print("\n=== Sayan project/app ===")
    sayan_app = list_files('sayan23bcy6/MKS_Hackathon', 'project/app')
    print(sayan_app)

    print("\n=== Divyanshu src ===")
    div_src = list_files('divyanshusinghtiwari/MKS-AI-Hackathon', 'src')
    print(div_src)

    # Inspect key files from Sayan
    for fname in ['config.py', 'clean.py', 'reconcile.py', 'features.py', 'models.py', 'backtest.py']:
        if fname in sayan_pipeline:
            content = get_file('sayan23bcy6/MKS_Hackathon', f'project/pipeline/{fname}')
            print(f"\n--- Sayan {fname} (first 600 chars) ---")
            print(content[:600])

    # Inspect key files from Divyanshu
    for fname in div_src:
        content = get_file('divyanshusinghtiwari/MKS-AI-Hackathon', f'src/{fname}')
        print(f"\n--- Divyanshu {fname} (first 400 chars) ---")
        print(content[:400])
