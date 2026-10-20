# Architecture

이 문서는 레포의 큰 구조를 설명합니다. 코드를 처음 열었을 때 "무엇이 어디 있고, 무엇이 무엇에 기대는지"를
알려 주는 지도입니다. 자주 바뀌지 않는 것만 적습니다. 함수 이름이나 파일은 링크 대신 이름으로 적었으니
검색해서 찾으세요. 전략 규칙 자체는 [docs/dongpa_strategy.md](docs/dongpa_strategy.md)에 있습니다.

## 조감도

동파법은 하루에 한 번, 장 마감 종가로 사고파는 전략입니다. 그래서 이 레포의 모든 것은
**"전일 종가까지의 기록 → 오늘 넣을 LOC 주문표"** 라는 계산 하나를 중심으로 돕니다.

```
             data/*.csv (일봉)      config/ (전략·계좌)
                    │                    │
                    ▼                    ▼
              ┌───────────────────────────────┐
              │ engines/  (순수 계산, I/O 없음) │  백테스트 · 주문표 · VR
              └───────────────────────────────┘
                 ▲            ▲             ▲
                 │            │             │
        benchmarks/      web/api/  ◀───── trader/
        (전략 연구)     (FastAPI)          (자동매매 잡)
                            │      실행 기록     │
                            ▼   ◀── GCS ───   ▼
                     web/frontend/          증권사 API
                     (React, 보기 전용)      (메리츠)
```

- **engines**가 유일한 계산 원천입니다. 백테스트와 오늘의 주문표는 같은 엔진을 같은 입력으로 돌린 결과입니다.
- **web**은 사람이 보는 창입니다. 계산을 요청하고 결과를 보여 줄 뿐, 주문을 내거나 설정을 바꾸지 않습니다.
- **trader**는 증권사와 말하는 유일한 곳입니다. 웹과 같은 주문표를 받아 실제 계좌와 맞춰 보고, 맞을 때만 주문합니다.
- **benchmarks**는 엔진에 다른 모드 신호를 넣어 보는 연구용 실험실입니다.

## 코드 지도

### `engines/` — 계산

전략의 모든 규칙이 여기 있습니다. pandas DataFrame과 설정을 받아 결과를 돌려주는 순수 함수만 둡니다.
네트워크, 파일 쓰기, 현재 시각을 읽지 않습니다.

- `dongpa_engine.py`: 백테스트 본체 `run_backtest()`. 모드 판단(주봉 RSI, `trend_vix`), 트랜치 예산,
  LOC 체결 규칙, 수수료, 일지·거래 기록. 설정 묶음은 `StrategyParams` / `ModeParams` / `CapitalParams`.
- `order_book_engine.py`: 백테스트가 남긴 미청산 lot으로 **다음 거래일 주문표**를 만듭니다.
  익절·만료 매도, 매수, 1주 사다리(`build_spread_orders`), 퉁치기(`apply_netting`).
- `vr_engine.py`: 라오어 VR. 동파법과 별개의 전략으로, 2주 사이클과 매수·매도 사다리를 계산합니다.
- `config_store.py`: `config/`를 읽는 유일한 곳. 쓰기 함수는 없습니다.

### `config/` — 무엇을 굴리는가

- `strategy.json`: 모든 계좌가 함께 쓰는 동파법 매매 숫자.
- `accounts/<id>.json`: 계좌별 시작일·시작 금액·사다리·수수료. 엔진은 시작일부터 다시 계산해 지금 장부를 얻습니다.
- `presets/`: 저장해 둔 전략 후보. `vr/`: VR 계좌.
- `trader/<id>.json`: 자동매매 설정(증권사, `auto_trade`). 비밀값은 없습니다.

개인 계좌 파일(`*.json`)은 gitignore이고 `*.sample`만 커밋됩니다.

### `data/` — 시세

일봉 OHLCV CSV(`data/tickers.txt`의 종목). `scripts/update_market_data.py`가 새 봉을 덧붙입니다.
읽는 쪽(`web/api/app/services/market_data.py`의 `MarketDataClient`)은 CSV를 먼저 쓰고, 마지막 행 이후만
Yahoo에서 받아 채웁니다. 장중에는 오늘의 임시 봉을 버립니다.

### `web/` — 보는 창

- `web/api/app/main.py`: FastAPI 라우트(`/api/v1/...`)와 빌드된 프런트엔드 서빙.
- `web/api/app/services/`: 라우트 뒤의 일. `backtest.py`(백테스트·주문표 화면 데이터, `run_order_book_view`),
  `vr.py`, `accounts.py`, `market_data.py`, `benchmarks.py`(저장된 결과 읽기), `trader.py`(자동매매 실행 기록 읽기).
- `web/frontend/`: React + TypeScript(Vite). 탭은 오더북 · 백테스트 · VR · 벤치마크 · 설정.
  오더북 맨 위 `AutoTradePanel.tsx`가 자동매매 기록을 가격 지도와 하루 타임라인으로 보여 줍니다.

### `trader/` — 자동매매

`python -m trader <step>`. 계좌마다 증권사 잔고를 엔진 장부와 맞춰 보고, 주문을 냅니다.

- `cli.py` / `runner.py`: 단계 실행. 단계마다 실행 기록 하나를 남깁니다.
  - `check`: 달력 + 대조만 (읽기 전용)
  - `orders`: 대조 → 주문표를 LOC/MOC로 전송
  - `stop`: 마감 직전 현재가로 손절 lot을 골라 익절 주문을 취소하고 MOC 매도
- `expected.py`: 엔진이 말하는 "계좌에 있어야 할 것". 웹의 `run_order_book_view`를 그대로 불러서,
  화면에 보이는 주문표와 다른 주문표를 계산할 수 없게 합니다.
- `plan.py`, `stop.py`, `reconcile.py`: 순수 함수. 주문표 행 → 주문, 손절 계획, 장부 대 잔고 비교.
- `calendar.py`: NYSE 휴장·조기 마감(`exchange_calendars`), 단계별 시각.
- `broker/`: `base.py`의 `Broker` 프로토콜과 중립 타입(Position, Order, LOC/MOC…). 증권사마다 어댑터 하나
  (`meritz.py`), `BROKERS`에 등록합니다. 증권사 고유의 코드·필드 이름은 어댑터 밖으로 나오지 않습니다.
- `runlog.py`: 실행 기록 저장소. 로컬(`var/trader`) 또는 GCS(`DONGPA_TRADER_RUNS`).
- `config.py`: `config/trader` 읽기와 증권사 키(환경변수 `<BROKER>_<ID>_APP_KEY`).

### `benchmarks/` — 연구

`python -m benchmarks <suite>`. 실제 엔진에 모드 신호를 주입해 전략 후보를 비교합니다(`lib/engine.py`).
suite 하나가 파일 하나(`suites/`)입니다. 결과 요약(`results/summary.json`)은 커밋되어 웹 벤치마크 탭이 읽습니다.
자세한 내용은 `benchmarks/README.md`.

### `infra/` 와 이미지

- `Dockerfile.web`: 웹 이미지(프런트 빌드 + API). Cloud Run 서비스, IAP 뒤.
- `Dockerfile.trader`: 자동매매 이미지(`config/trader` 포함, 키 없음). Cloud Run Job 두 개
  (`orders`, `stop`)를 Cloud Scheduler가 뉴욕 시간으로 깨웁니다.
- `infra/__main__.py`: Pulumi로 관리하는 GCP 리소스 전부(서비스, 잡, 스케줄러, 버킷, IAM). 개인 값은 `.env`에서 옵니다.
- `Makefile`: `make deploy`(웹 이미지), `make trader-deploy`(자동매매 이미지), `make trader-secrets`(키 → Secret Manager),
  `make infra-preview` / `infra-up`.

### `tests/`

pytest. 엔진 규칙(`test_loc_rules.py`, `test_order_book_engine.py` …), 웹 API, 자동매매(가짜 증권사와 가짜 HTTP)를
다룹니다. 네트워크와 실제 계좌는 쓰지 않습니다.

## 하루의 흐름

모든 시각은 뉴욕 시간입니다.

| 시각 | 무엇이 | 하는 일 |
|---|---|---|
| 전일 마감 후 | 시세 | 전일 종가가 생김. 다음 계산부터 엔진이 그 봉까지 봄 |
| 09:45 | `orders` 잡 | 달력 확인 → 엔진 장부와 증권사 잔고 대조 → 맞으면 주문표 전송 |
| 장중 | 웹 | 사람이 주문표·자동매매 기록을 봄 (언제 열어도 같은 주문표) |
| 마감 15분 전 | `stop` 잡 | 현재가 ≤ 손절가인 lot의 익절 LOC 취소 → MOC 매도 |
| 마감 10분 전 | 증권사 | LOC·MOC 접수 마감 |
| 마감 | 증권사 | LOC는 종가가 조건에 맞으면 종가에 체결. 엔진도 같은 규칙으로 그날을 계산 |
| 다음 날 09:45 | `orders` 잡 | 체결 결과가 엔진 계산과 같았는지 대조로 확인. 어긋났으면 멈춤 |

엔진은 실제 체결을 입력으로 받지 않습니다. 실제 계좌가 엔진 계산을 따라가고, 매일 아침 대조가 그걸 검증합니다.

## 경계와 불변식

여기 적힌 것이 깨지면 구조가 깨진 것입니다.

- **계산은 한 곳.** 주문표는 `engines/order_book_engine.py`에서만 만들어집니다. 웹 화면, PiP 창, 자동매매 모두 같은 함수
  (`run_order_book_view`)의 결과를 씁니다. 다른 곳에서 가격이나 수량을 다시 계산하지 않습니다.
- **engines는 I/O를 하지 않습니다.** 시세, 설정, 현재 시각은 모두 바깥에서 넣어 줍니다. 그래서 같은 입력이면 언제나 같은 결과입니다.
- **web은 보기 전용입니다.** 설정을 쓰지 않고, 주문을 내지 않고, 증권사 설정이나 키를 모릅니다. 자동매매는 실행 기록으로만 봅니다.
  설정을 바꾸려면 파일을 고치고 다시 배포합니다.
- **증권사와 말하는 것은 `trader/broker/`뿐입니다.** 위쪽 코드는 `Broker` 프로토콜과 중립 타입만 압니다.
- **의심스러우면 멈춥니다.** 달력이 다르거나, 시세가 늦거나, 장부와 잔고가 다르거나, 이미 낸 주문이 있으면 주문하지 않고
  `halted`로 기록합니다. 응답을 못 받은 주문은 다시 보내지 않습니다(접수됐을 수 있음). 같은 날 두 번 실행해도 두 번 주문하지 않습니다.
- **모든 실행은 기록을 남깁니다.** 아무 일도 안 한 날(`closed`)도 남깁니다. 기록 저장에 실패하면 로컬에라도 남깁니다.
- **비밀값은 레포와 이미지에 없습니다.** 로컬은 `.env`, 클라우드는 Secret Manager → 환경변수. Pulumi 상태에도 들어가지 않습니다.

## 여러 곳에 걸친 것

- **날짜.** 모든 날짜는 미국 장 날짜(뉴욕)입니다. 서버의 "오늘"은 `market_today()`, 자동매매는 `ny_now()`입니다.
  한국 시간으로 바꾸지 않습니다.
- **돈과 수량.** 금액은 `Decimal`로 센트 단위 반올림(`money()`, `to_decimal()`), 주식 수는 정수입니다. 주문가도 같은 규칙으로 반올림해서
  엔진과 증권사 주문이 1센트도 다르지 않게 합니다.
- **설정.** 사람이 바꾸는 것은 모두 `config/`의 JSON입니다. 읽는 곳은 `engines/config_store.py`와 `trader/config.py`뿐입니다.
- **시각과 달력.** 휴장·조기 마감은 `trader/calendar.py`가 판단하고, 증권사의 장 운영 여부와 둘 다 맞아야 실행합니다.
  스케줄러는 깨우기만 하고, 지금이 그 단계의 시간인지는 각 단계가 정합니다.
- **테스트.** 순수한 부분(엔진, 주문표, 대조, 계획)은 단위 테스트로, I/O 부분은 가짜 증권사·가짜 HTTP로 시험합니다.
  엔진을 바꾸면 `benchmarks`의 `verify_harness`, `verify_trend_vix`도 돌립니다.

## 더 읽을 것

- [docs/dongpa_strategy.md](docs/dongpa_strategy.md): 전략 규칙
- [docs/loc_netting.md](docs/loc_netting.md), [docs/loc_spread_buy_strategy.md](docs/loc_spread_buy_strategy.md): 퉁치기, 1주 사다리
- [docs/vr_sheet.md](docs/vr_sheet.md): VR 시트 규칙
- [benchmarks/README.md](benchmarks/README.md): 전략 연구
- [infra/README.md](infra/README.md): GCP 리소스와 처음 설정
