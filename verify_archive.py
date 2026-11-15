from pathlib import Path
import ast,hashlib,json
r=Path(__file__).resolve().parent
for x in json.loads((r/'MANIFEST.json').read_text(encoding='utf-8')):
 p=r/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],str(p)
count=0
for p in r.rglob('*.py'):
 ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p));count+=1
print('Archive hash and syntax checks passed:',count,'Python files')
