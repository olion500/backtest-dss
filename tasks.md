# Tasks

## In Progress
- [ ] 무료 호스팅 배포 (후보: 집 PC + Cloudflare Tunnel / Render free)

## Done
- [x] yfinance 의존 제거: 일봉 데이터를 git으로 관리 (2026-09-06)
  - `scripts/update_market_data.py` 증분 다운로드 (7일 겹침 재수집, 3회 재시도)
  - `.github/workflows/update-data.yml` 평일 22:30 UTC cron → data/ 커밋
  - 뷰어는 dataset 우선 (로컬 data/ → raw GitHub), 미추적 티커·7일 이상 낡은 데이터만 yfinance 폴백
- [x] 웹 뷰어 (FastAPI + React) 커밋·푸시 (2026-09-06)
- [x] 오더북 진입 시 자동 실행, 계산 버튼 제거 (2026-09-06)
- [x] 넓은 화면에서 본문 가운데 정렬 (2026-09-06)
