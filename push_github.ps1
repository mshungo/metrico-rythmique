param(
    [string]$Message = "Update TextAnalysis platform"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  GitHub (mshungo/metrico-rythmique) 自動プッシュ & Streamlit 同期" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host "`n1. ファイルのステージング中..." -ForegroundColor Yellow
git add .

$status = git status --porcelain
if (-not $status) {
    Write-Host "ローカルの未コミット変更はありません。リモート同期と公開状態を確認します..." -ForegroundColor Green
} else {
    Write-Host "`n2. コミットを作成中: '$Message'..." -ForegroundColor Yellow
    git commit -m $Message
}

Write-Host "`n3. GitHubへプッシュ中 (origin main)..." -ForegroundColor Yellow
git push origin main

Write-Host "`n4. https://metrythme.streamlit.app/ の稼働・同期状態を確認中..." -ForegroundColor Yellow
python -c @"
import urllib.request, urllib.error, http.cookiejar, json, time
APP_URL = 'https://metrythme.streamlit.app'
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.open(APP_URL + '/', timeout=30)
resp = opener.open(APP_URL + '/api/v2/app/status', timeout=30)
csrf = resp.headers.get('x-csrf-token', '')
data = json.loads(resp.read().decode('utf-8'))
st_code = data.get('status')
if st_code == 12 and csrf:
    print('スリープ状態を検出しました。自動復帰を実行します...')
    req = urllib.request.Request(APP_URL + '/api/v2/app/resume', data=b'{}', headers={'x-csrf-token': csrf, 'Content-Type': 'application/json'}, method='POST')
    opener.open(req, timeout=30)
for i in range(25):
    try:
        st_code = json.loads(opener.open(APP_URL + '/api/v2/app/status', timeout=30).read().decode('utf-8')).get('status')
        if st_code == 5:
            health = opener.open(APP_URL + '/~/+/_stcore/health', timeout=30).read().decode('utf-8').strip()
            if health == 'ok':
                print(f'Streamlit Cloud 稼働確認完了: status={st_code} (RUNNING), health={health}')
                break
    except urllib.error.HTTPError:
        pass
    time.sleep(3)
"@

Write-Host "`n>>> GitHubへのプッシュおよび https://metrythme.streamlit.app/ の公開確認が完了しました！" -ForegroundColor Green
