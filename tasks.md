# Tasks

## In Progress
- [ ] 데이터 자동 갱신 방법 결정: GitHub Actions 러너(Azure IP)가 Yahoo에 차단됨
  - 후보 A: 집 PC 스케줄러(Windows Task Scheduler → WSL)에서 fetch + push
  - 후보 B: Tiingo 무료 API 키 발급 후 Actions에서 Tiingo로 fetch
  - 워크플로 cron은 임시 비활성화 (workflow_dispatch만 유지)
- [ ] Cloud Run 배포 (scale-to-zero, asia-northeast1 도쿄 = Tier 1 무료 한도)
  - Phase 0 (사용자): GCP 가입·카드, Budget $5 알림, gcloud auth login
  - Phase 1: entrypoint `--port ${PORT:-8000}` (완료). boot 백그라운드 데이터 갱신은 유지
    - Cloud Run 참고: 요청 기반 과금에선 요청 처리 중에만 CPU가 배정되어 백그라운드 갱신이 느리거나 멈출 수 있음.
      실패해도 베이스라인 + 갭필로 정상 동작하므로 문제 시에만 startup CPU boost 검토
  - Phase 2 (완료 2026-09-10): personal-447204 프로젝트, asia-northeast1
    - https://dongpa-viewer-138405702335.asia-northeast1.run.app (1Gi/1cpu, max-instances 1)
    - 백테스트·오더북 API 정상 (1.7s/0.2s), 예산 ₩10,000/월 알림 기존재, 이미지 최근 2개 보관 정책 설정
    - IAP 구글 로그인 설정 (2026-09-10): 조직 없는 개인 프로젝트라 관리형 클라이언트 불가 →
      커스텀 OAuth 클라이언트(dongpa-iap) 생성 후 `gcloud iap settings set`으로 연결.
      허용 계정: ymj02349@gmail.com (iap.httpsResourceAccessor). make deploy에 --iap 반영
    - 콜드스타트 실측은 IAP 로그인 후 브라우저에서 체감 확인으로 대체
  - Phase 3: WIF 인증 + deploy-web.yml (web/·engines/·Dockerfile.web 변경 시), 이미지 2개만 보관
  - Phase 4: 데이터 갱신 재개되면 DONGPA_DATA_URL=raw GitHub 전환, (선택) 커스텀 도메인
  - 429 심하면 대안: Render free / 집 PC + Cloudflare Tunnel (조사 문서: claude.ai/code/artifact/fe573437-3a8e-4882-a918-c473ea653852)

## Done
- [x] yfinance 의존 제거: 일봉 데이터를 git으로 관리 (2026-09-06)
  - `scripts/update_market_data.py` 증분 다운로드 (7일 겹침 재수집, 3회 재시도)
  - `.github/workflows/update-data.yml` 평일 22:30 UTC cron → data/ 커밋
  - 뷰어는 dataset 우선 (로컬 data/ → raw GitHub), 미추적 티커·7일 이상 낡은 데이터만 yfinance 폴백
- [x] 웹 뷰어 (FastAPI + React) 커밋·푸시 (2026-09-06)
- [x] 오더북 진입 시 자동 실행, 계산 버튼 제거 (2026-09-06)
- [x] 넓은 화면에서 본문 가운데 정렬 (2026-09-06)
