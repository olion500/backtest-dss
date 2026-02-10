# Code Review TODO

## Critical Bugs

- [x] **BUG-1**: `backtest.py:490` - 데이터 비어있을 때 `st.stop()` 누락으로 NameError 크래시. `st.error()` 뒤에 `st.stop()` 추가.
- [x] **BUG-2**: `dongpa_engine.py:399-404` - `reset_on_mode_change` 플래그 무효. dead branch 제거, 필드 및 모든 호출부에서 완전 삭제. 문서 업데이트.

## High Priority Bugs

- [x] **BUG-3**: `run_optuna.py:81` - `apply_to_config()`에서 `mode_switch_strategy_index`를 실제 전략에 맞게 수정. RSI 임계값/MA 기간도 저장하도록 추가.
- [x] **BUG-4**: `backtest.py:570` - 모멘텀 Buy & Hold를 사용자 지정 기간(`start`~`end`)으로 슬라이스하여 계산하도록 수정.
- [x] **BUG-5**: `pages/2_orderBook.py:674,738` - 잔여일을 엔진의 `보유기간(일)` (거래일 기준)으로 계산하도록 변경. 캘린더일 방식 제거.

## Logic Issues

- [x] **LOGIC-1**: `dongpa_engine.py:512-513` - MOC hold_days dead code 제거. `min(hold_days + 1, max_hold)` 항상 무효였음.
- [x] **LOGIC-2**: `dongpa_engine.py:536-537` - tranche_base_cash를 매도 후 항상 현재 cash로 리셋하도록 변경. 이전 ratchet(비감소) 방식 제거. budget deadlock 해소.
- [x] **LOGIC-3**: `dongpa_engine.py:558-567` - sell_summary 라벨을 "매도"에서 "TP대기"로 변경하여 미래 매도 대기 주문임을 명시.
- [x] **LOGIC-4**: Netting 코드에 display-only임을 명시하는 코멘트 추가. 실제 cash/lots에 영향 없음을 문서화.

## Refactoring

- [ ] **REFACTOR-1**: `render_navigation()` 3곳 복사붙이기 (`backtest.py`, `orderBook.py`, `3_Optuna.py`). 공통 모듈로 추출.
- [ ] **REFACTOR-2**: `compute_trade_metrics()` (`backtest.py:133`) vs `_compute_metrics()` (`orderBook.py:224`) 거의 동일. 통합.
- [ ] **REFACTOR-3**: `_determine_initial_mode()` RSI 로직 (`dongpa_engine.py:293-345`)이 `_decide_mode_rsi()` (233-263) 중복 복사. 변경 시 두 곳 동시 수정 필요.
- [ ] **REFACTOR-4**: `apply_to_config()` 중복 - `run_optuna.py:62` vs `pages/3_Optuna.py:207`. 통합 필요 (3_Optuna.py 버전이 올바름).
- [ ] **REFACTOR-5**: 기본값 (`SOXL`, `QQQ`, `SOXX`, 10000, 7 slices 등) 5개 파일에 분산 하드코딩. 단일 config로 통합.
- [ ] **REFACTOR-6**: `_load_settings()` 중복 - `backtest.py:55` vs `orderBook.py:55`. 공통 함수로 추출.

## Dead Code / Unused

- [ ] **DEAD-1**: `CapitalParams.slippage_pct` - 정의만 있고 엔진에서 한 번도 사용 안 됨. 제거하거나 실제 구현.
- [ ] **DEAD-2**: `StrategyParams.benchmark_ticker` - 엔진 내부에서 미사용. UI에서만 사용. StrategyParams에서 제거 고려.
- [ ] **DEAD-3**: `create_objective()`에 `combined_target/combined_momo` 파라미터 전달되지만 objective 내부에서 미사용.

## Performance

- [ ] **PERF-1**: `dongpa_optuna.py:455` - `extract_results()`에서 Yahoo Finance 재다운로드. `run_optuna()`의 데이터를 재사용하도록 변경.
- [ ] **PERF-2**: `pages/2_orderBook.py` - 페이지 로드마다 전체 백테스트 실행. `st.cache_data` 캐싱 없음.

## Type Safety

- [ ] **TYPE-1**: `StrategyParams.defense`/`offense` 기본값 None. None 검증 없이 속성 접근하면 AttributeError.
- [ ] **TYPE-2**: `OptunaConfig.constraints` 타입 `list[...]`이지만 기본값 `None`. `list | None` 또는 `field(default_factory=list)`.
