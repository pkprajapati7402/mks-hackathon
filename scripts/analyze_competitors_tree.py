import urllib.request
import json

def list_tree(repo, path=''):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            for item in data:
                print(f"{item['type']}: {item['path']}")
                if item['type'] == 'dir' and path.count('/') < 2:
                    list_tree(repo, item['path'])
    except Exception as e:
        print('Error:', e)

print('=== Sayan Tree ===')
list_tree('sayan23bcy6/MKS_Hackathon', 'project')
print('\n=== Divyanshu Tree ===')
list_tree('divyanshusinghtiwari/MKS-AI-Hackathon', 'src')
