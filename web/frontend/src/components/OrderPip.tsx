import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { drawCard, VIDEO_CARD_SIZE } from "../pip/drawCard";
import { buildSteps, ladderLabel, orderCount, orderPriceText, priceText, qtyText, stepSide, type PipStep } from "../pip/steps";
import type { OrderBookPayload } from "../types";

// 주문을 한 건씩 띄워 MTS에 보고 따라 입력하는 창. 브라우저가 가진 기능에 따라 방식을 고른다.
// - document: Document Picture-in-Picture (Chrome·Edge 데스크톱). 창 안에 HTML과 버튼을 그대로 넣는다
// - video: 카드를 canvas에 그려 영상 PiP로 띄운다 (안드로이드 Chrome 등). 넘기기는 시스템 이전·다음 트랙 버튼
// - overlay: 둘 다 없으면(삼성 인터넷, iOS Safari) 페이지 전체에 띄우고 OS 팝업·분할 화면으로 MTS 옆에 둔다
type PipMode = "document" | "video" | "overlay";

interface DocumentPictureInPicture {
  requestWindow(options?: { width?: number; height?: number }): Promise<Window>;
  window: Window | null;
}

declare global {
  interface Window {
    documentPictureInPicture?: DocumentPictureInPicture;
  }
}

const PIP_SIZE = { width: 400, height: 300 };

function detectMode(): PipMode {
  if ("documentPictureInPicture" in window) return "document";
  const videoPip = document.pictureInPictureEnabled && typeof HTMLVideoElement.prototype.requestPictureInPicture === "function";
  if (videoPip && typeof HTMLCanvasElement.prototype.captureStream === "function") return "video";
  return "overlay";
}

function loadIndex(key: string) {
  try {
    const value = Number(localStorage.getItem(key));
    return Number.isInteger(value) && value > 0 ? value : 0;
  } catch {
    return 0;
  }
}

function saveIndex(key: string, index: number) {
  try {
    localStorage.setItem(key, String(index));
  } catch {
    // storage unavailable: progress lasts until the window closes
  }
}

/** 메인 문서의 스타일시트를 PiP 창으로 옮긴다 (dev는 <style>, 빌드는 <link>). */
function copyStyles(target: Document) {
  for (const sheet of Array.from(document.styleSheets)) {
    try {
      const style = target.createElement("style");
      style.textContent = Array.from(sheet.cssRules, (rule) => rule.cssText).join("\n");
      target.head.appendChild(style);
    } catch {
      if (!sheet.href) continue;
      const link = target.createElement("link");
      link.rel = "stylesheet";
      link.href = sheet.href;
      target.head.appendChild(link);
    }
  }
}

function Chevron({ dir }: { dir: "prev" | "next" }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden>
      <path d={dir === "next" ? "M9 5l7 7-7 7" : "M15 5l-7 7 7 7"} fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="square" />
    </svg>
  );
}

function PipStepper({
  steps,
  index,
  onMove,
  onReset,
}: {
  steps: PipStep[];
  index: number;
  onMove: (delta: number) => void;
  onReset: () => void;
}) {
  if (!steps.length) {
    return <div className="opip-end"><strong>주문 없음</strong><span>오늘 넣을 LOC 주문이 없습니다</span></div>;
  }
  if (index >= steps.length) {
    return (
      <div className="opip-end">
        <strong>오늘 주문 끝</strong>
        <span>{orderCount(steps)}건을 모두 넣었습니다</span>
        <button type="button" className="opip-back opip-reset" onClick={onReset}>처음부터</button>
      </div>
    );
  }

  const step = steps[index];
  const last = index === steps.length - 1;
  return (
    <div className="opip-stage">
      <div className="opip-segs" aria-hidden>
        {steps.map((s, k) => <i key={k} className={`${stepSide(s)}${k < index ? " done" : ""}${k === index ? " cur" : ""}`} />)}
      </div>
      <div className="opip-meta">
        {step.kind === "single"
          ? <span className={`opip-tag ${step.order.side}`}>{step.order.label}</span>
          : <span className="opip-tag buy">매수 스프레드 {step.orders.length}건</span>}
        <span className="opip-count">{index + 1} / {steps.length}</span>
      </div>

      {step.kind === "single" ? (
        <div className="opip-main">
          <span className={`opip-price ${step.order.side}`}>{orderPriceText(step.order)}</span>
          <span className="opip-qty">{qtyText(step.order.qty)}<small>주</small></span>
          {/* 설명이 없어도 줄을 남겨 매수·매도의 숫자 위치를 맞춘다 */}
          <span className="opip-note">{step.order.note || " "}</span>
        </div>
      ) : (
        <div className="opip-ladder">
          {step.orders.map((order, k) => (
            <div key={k} className="opip-ladder-row">
              <span className="opip-lbl">{ladderLabel(order)}</span>
              <span className="opip-ladder-price">{priceText(order.price)}</span>
              <span className="opip-ladder-qty">× {qtyText(order.qty)}주</span>
            </div>
          ))}
        </div>
      )}

      <div className="opip-nav">
        <button type="button" className="opip-back" disabled={index === 0} aria-label="이전 주문" onClick={() => onMove(-1)}><Chevron dir="prev" /></button>
        <button type="button" className="opip-go" onClick={() => onMove(1)}>
          {step.kind === "ladder" ? "다 넣었음" : "넣었음"} · {last ? "끝" : "다음"}<Chevron dir="next" />
        </button>
      </div>
    </div>
  );
}

export function OrderPipButton({
  orderBook,
  ticker,
  accountId,
}: {
  orderBook: OrderBookPayload;
  ticker: string;
  accountId?: string;
}) {
  const [mode] = useState(detectMode);
  const [pipWindow, setPipWindow] = useState<Window | null>(null);
  const [videoOpen, setVideoOpen] = useState(false);
  const [overlay, setOverlay] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const steps = useMemo(() => buildSteps(orderBook.orders), [orderBook.orders]);

  // 진행 위치는 계좌·세션 날짜·주문 내용별로 저장한다. 주문표가 바뀌면 처음부터 다시 시작한다.
  const progressKey = useMemo(() => {
    const signature = orderBook.orders.rows.map((row) => `${row["구분"]}@${row["주문가"]}x${row["수량"]}`).join("|");
    return `dongpa-pip-progress:${accountId ?? "local"}:${ticker}:${orderBook.state.last_date}:${signature}`;
  }, [accountId, ticker, orderBook.state.last_date, orderBook.orders.rows]);

  const [progress, setProgress] = useState(() => ({ key: progressKey, index: loadIndex(progressKey) }));
  const index = progress.key === progressKey ? progress.index : loadIndex(progressKey);

  const setIndex = useCallback((next: number) => {
    const clamped = Math.max(0, Math.min(steps.length, next));
    saveIndex(progressKey, clamped);
    setProgress({ key: progressKey, index: clamped });
  }, [progressKey, steps.length]);

  const move = useCallback((delta: number) => setIndex(index + delta), [index, setIndex]);
  // 미디어 세션 핸들러는 한 번만 등록하므로 최신 move를 ref로 넘긴다
  const moveRef = useRef(move);
  useEffect(() => { moveRef.current = move; }, [move]);

  useEffect(() => {
    const target = pipWindow ?? (overlay ? window : null);
    if (!target) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight" || event.key === " ") {
        event.preventDefault();
        move(1);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        move(-1);
      } else if (event.key === "Escape" && !pipWindow) {
        setOverlay(false);
      }
    };
    target.addEventListener("keydown", onKey);
    return () => target.removeEventListener("keydown", onKey);
  }, [pipWindow, overlay, move]);

  // 전체 화면 모드에서는 뒤 페이지가 스크롤되지 않게 막는다.
  useEffect(() => {
    if (!overlay) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previous; };
  }, [overlay]);

  // 오더북 화면이 사라지면(다시 계산, 다른 탭) PiP 창도 닫는다.
  useEffect(() => () => pipWindow?.close(), [pipWindow]);
  useEffect(() => {
    const video = videoRef.current;
    return () => {
      if (video && document.pictureInPictureElement === video) void document.exitPictureInPicture();
    };
  }, []);

  // --- 영상 PiP ---
  const redraw = useCallback(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) drawCard(ctx, steps, index);
  }, [steps, index]);

  useEffect(() => {
    if (mode !== "video") return;
    // canvas 글꼴이 대체 글꼴로 그려지지 않게 미리 받아 둔다
    void document.fonts?.load("40px Juache").then(redraw, () => undefined);
  }, [mode, redraw]);

  useEffect(() => {
    if (!videoOpen) return;
    redraw();
    // 일부 브라우저는 canvas가 다시 그려질 때만 프레임을 보내므로 주기적으로 다시 그린다
    const timer = window.setInterval(redraw, 500);
    return () => window.clearInterval(timer);
  }, [videoOpen, redraw]);

  useEffect(() => {
    const video = videoRef.current;
    if (mode !== "video" || !video) return;
    const onLeave = () => {
      setVideoOpen(false);
      video.pause();
    };
    video.addEventListener("leavepictureinpicture", onLeave);
    return () => video.removeEventListener("leavepictureinpicture", onLeave);
  }, [mode]);

  useEffect(() => {
    if (!videoOpen || !("mediaSession" in navigator)) return;
    const session = navigator.mediaSession;
    session.metadata = new MediaMetadata({ title: `${ticker} LOC 주문`, artist: "DONGPA" });
    session.setActionHandler("previoustrack", () => moveRef.current(-1));
    session.setActionHandler("nexttrack", () => moveRef.current(1));
    return () => {
      session.setActionHandler("previoustrack", null);
      session.setActionHandler("nexttrack", null);
      session.metadata = null;
    };
  }, [videoOpen, ticker]);

  const openVideo = async () => {
    const video = videoRef.current;
    if (!video) return;
    if (document.pictureInPictureElement === video) {
      await document.exitPictureInPicture();
      return;
    }
    try {
      let canvas = canvasRef.current;
      if (!canvas) {
        canvas = document.createElement("canvas");
        canvas.width = VIDEO_CARD_SIZE.width;
        canvas.height = VIDEO_CARD_SIZE.height;
        canvasRef.current = canvas;
      }
      const ctx = canvas.getContext("2d");
      if (ctx) drawCard(ctx, steps, index);
      if (!video.srcObject) video.srcObject = canvas.captureStream(15);
      await video.play();
      if (video.readyState < HTMLMediaElement.HAVE_METADATA) {
        await new Promise((resolve) => video.addEventListener("loadedmetadata", resolve, { once: true }));
      }
      await video.requestPictureInPicture();
      setError(null);
      setVideoOpen(true);
    } catch (reason) {
      setError(`PiP를 열지 못했습니다: ${reason instanceof Error ? reason.message : String(reason)}`);
    }
  };

  // --- Document PiP ---
  const openDocument = async () => {
    const api = window.documentPictureInPicture;
    if (!api) return;
    if (pipWindow) {
      pipWindow.close();
      return;
    }
    const win = await api.requestWindow(PIP_SIZE);
    copyStyles(win.document);
    win.document.title = `${ticker} LOC 주문`;
    win.document.body.className = "opip-root";
    win.addEventListener("pagehide", () => setPipWindow(null));
    setPipWindow(win);
  };

  const stepper = <PipStepper steps={steps} index={index} onMove={move} onReset={() => setIndex(0)} />;

  if (mode === "overlay") {
    return (
      <>
        <button type="button" className="button pip-launch" onClick={() => setOverlay(true)} title="주문을 한 건씩 화면 가득 띄웁니다">
          주문 넘기기
        </button>
        {overlay && createPortal(
          <div className="opip-overlay" role="dialog" aria-label={`${ticker} LOC 주문`}>
            <header className="opip-overlay-bar">
              <span>{ticker} · LOC 주문</span>
              <button type="button" onClick={() => setOverlay(false)}>닫기</button>
            </header>
            <div className="opip-overlay-body">{stepper}</div>
          </div>,
          document.body,
        )}
      </>
    );
  }

  if (mode === "video") {
    return (
      <>
        <button type="button" className={`button pip-launch${videoOpen ? " active" : ""}`} onClick={openVideo} title="주문을 영상 PiP 창에 띄웁니다. 창의 이전·다음 버튼으로 넘깁니다">
          {videoOpen ? "PIP 닫기" : "PIP로 띄우기"}
        </button>
        {error && <span className="pip-error">{error}</span>}
        <video
          ref={videoRef}
          className="pip-video-source"
          muted
          playsInline
          aria-hidden
        />
      </>
    );
  }

  return (
    <>
      <button type="button" className={`button pip-launch${pipWindow ? " active" : ""}`} onClick={openDocument} title="항상 위에 떠 있는 작은 창에 주문을 한 건씩 띄웁니다">
        {pipWindow ? "PIP 닫기" : "PIP로 띄우기"}
      </button>
      {pipWindow && createPortal(stepper, pipWindow.document.body)}
    </>
  );
}
