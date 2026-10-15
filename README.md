# 동파법 백테스트 · 주문표

레버리지 ETF(SOXL 등)를 **동파법**으로 굴릴 때 쓰는 개인용 웹 도구입니다.
과거 성과를 돌려 보고, 오늘 넣을 주문을 계산해 줍니다.

## 할 수 있는 것

- **오더북**: 계좌의 매수 기록을 이어받아 오늘 넣을 LOC 주문표를 만듭니다. PiP 창으로 띄워 놓고 주문할 수 있습니다.
- **백테스트**: 기간과 매매 숫자를 바꿔 수익률(CAGR), 최대 낙폭(MDD), 매매 일지를 봅니다.
- **VR**: 라오어 밸류 리밸런싱 계좌의 이번 사이클 목표값과 매수·매도표를 봅니다. 구글 시트를 대신합니다.
- **벤치마크**: 새 전략 아이디어를 실제 엔진으로 검증합니다 ([benchmarks/](benchmarks/README.md)).

## 동파법 한눈에 보기

- 돈을 N개로 나눠(**트랜치**) 하루에 한 번만 삽니다.
- 주문은 모두 **LOC**(Limit-On-Close, 장 마감 지정가)입니다. 종가가 주문가 이하면 종가에 사고, 이상이면 안 삽니다.
- 산 물량마다 익절가가 있습니다. 종가가 익절가에 닿거나, 보유 기한이 끝나거나, 손절가 아래로 내려가면 팝니다.
- 시장 상황에 따라 **안전 모드**와 **공세 모드**를 오갑니다. 모드마다 분할 수·매수가·익절가가 다릅니다.
  - 기본: QQQ 주봉 RSI의 방향으로 고릅니다.
  - 대안: QQQ 이동평균 추세 + VIX로 고릅니다 (`trend_vix`).

자세한 규칙은 [docs/dongpa_strategy.md](docs/dongpa_strategy.md)에 있습니다.

## 빠른 시작

[uv](https://docs.astral.sh/uv/)와 Docker가 있으면 됩니다.

```bash
make web        # 이미지를 만들고 실행합니다
```

브라우저에서 http://localhost:8000 을 엽니다.

코드를 고치면서 보려면 터미널 두 개에서 실행합니다.

```bash
make install web-install   # 처음 한 번
make web-api               # API  → http://localhost:8000
make web-frontend          # 화면 → http://localhost:5173
```

## 내 계좌 넣기

설정 파일은 `config/`에 있습니다. 계좌 파일은 git에 올라가지 않습니다.

| 파일 | 내용 |
|---|---|
| `config/strategy.json` | 모든 계좌가 같이 쓰는 매매 숫자 |
| `config/accounts/<이름>.json` | 계좌별 시작일, 시작 금액, 스프레드 매수 설정. `example.json.sample`을 복사해서 만듭니다 |
| `config/vr/<이름>.json` | VR 계좌. `example.json.sample`을 복사해서 만듭니다 |
| `config/presets/` | 저장해 둔 다른 전략. 웹에서 골라 백테스트할 수 있습니다 |

웹 화면에서 바꾼 값은 그 브라우저에만 저장됩니다. 서버의 설정 파일은 바뀌지 않습니다.

## 폴더 구조

| 폴더 | 내용 |
|---|---|
| `engines/` | 계산 엔진: 백테스트, 주문표, VR |
| `web/` | 웹 앱: API(FastAPI)와 화면(React) |
| `benchmarks/` | 전략 연구용 벤치마크 |
| `data/` | 일봉 가격 데이터 (`scripts/update_market_data.py`가 갱신) |
| `config/` | 전략과 계좌 설정 |
| `docs/` | 전략·주문 규칙 문서 |
| `tests/` | 테스트 (`make test`) |

## 배포

Cloud Run에 올립니다. `.env.example`을 `.env`로 복사해 GCP 프로젝트를 적은 뒤 실행합니다.

```bash
make deploy
```

## 알아 둘 점

- 체결은 종가에 정확히 된다고 가정합니다. 슬리피지와 수수료는 계산하지 않습니다.
- 과거 성과가 미래 수익을 보장하지 않습니다. 투자 판단은 본인 책임입니다.

## 더 읽을거리

- [docs/dongpa_strategy.md](docs/dongpa_strategy.md): 전략 규칙과 파라미터
- [docs/loc_netting.md](docs/loc_netting.md): 같은 날 매수·매도 주문 상쇄 (퉁치기)
- [docs/loc_spread_buy_strategy.md](docs/loc_spread_buy_strategy.md): 스프레드 매수
- [docs/vr_sheet.md](docs/vr_sheet.md): VR 구글 시트 규칙과 웹 VR 탭
- [benchmarks/README.md](benchmarks/README.md): 벤치마크 실행·추가 방법과 연구 결과
