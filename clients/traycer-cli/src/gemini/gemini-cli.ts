import { spawn } from "node:child_process";
import { StringDecoder } from "node:string_decoder";
import type { Readable } from "node:stream";

export const GEMINI_APPROVAL_MODES = [
  "default",
  "plan",
  "auto_edit",
  "yolo",
] as const;

export type GeminiApprovalMode = (typeof GEMINI_APPROVAL_MODES)[number];

export interface GeminiCliInvocation {
  readonly binary: string;
  readonly prompt: string;
  readonly workingDirectory: string;
  readonly model: string | null;
  readonly approvalMode: GeminiApprovalMode;
  readonly resume: string | null;
  readonly sessionId: string | null;
  readonly skipTrust: boolean;
}

export interface GeminiStreamEvent {
  readonly kind: "event";
  readonly type: string;
  readonly raw: Readonly<Record<string, unknown>>;
  readonly rawLine: string;
}

export interface GeminiMalformedStreamLine {
  readonly kind: "malformed";
  readonly rawLine: string;
  readonly reason: string;
}

export type GeminiStreamItem = GeminiStreamEvent | GeminiMalformedStreamLine;

export interface GeminiCliRunHooks {
  readonly onEvent: (event: GeminiStreamEvent) => void;
  readonly onMalformedLine: (line: GeminiMalformedStreamLine) => void;
  readonly onStderr: (text: string) => void;
  readonly onCancellationRequested: (signal: NodeJS.Signals) => void;
}

export interface GeminiCliRunOutcome {
  readonly status: "success" | "error" | "cancelled";
  readonly exitCode: number;
  readonly signal: NodeJS.Signals | null;
  readonly sessionId: string | null;
  readonly model: string | null;
  readonly events: number;
  readonly malformedLines: number;
  readonly resultError: string | null;
}

export interface GeminiProcess {
  readonly pid?: number;
  readonly stdout: Readable | null;
  readonly stderr: Readable | null;
  once(event: "error", listener: (error: Error) => void): this;
  once(
    event: "close",
    listener: (code: number | null, signal: NodeJS.Signals | null) => void,
  ): this;
  kill(signal: NodeJS.Signals | undefined): boolean;
}

export interface GeminiProcessSpawner {
  spawn(
    command: string,
    args: readonly string[],
    options: {
      readonly cwd: string;
      readonly detached: boolean;
      readonly env: NodeJS.ProcessEnv;
      readonly stdio: ["ignore", "pipe", "pipe"];
      readonly windowsHide: boolean;
    },
  ): GeminiProcess;
}

export interface GeminiSignalProcess {
  readonly platform: NodeJS.Platform;
  kill(pid: number, signal: NodeJS.Signals): void;
  on(signal: NodeJS.Signals, listener: () => void): this;
  off(signal: NodeJS.Signals, listener: () => void): this;
}

export interface GeminiCliRunDependencies {
  readonly signalProcess: GeminiSignalProcess;
  readonly spawner: GeminiProcessSpawner;
  readonly setTimeout: (
    callback: () => void,
    delayMs: number,
  ) => NodeJS.Timeout;
  readonly clearTimeout: (timeout: NodeJS.Timeout) => void;
}

const CANCELLATION_KILL_GRACE_MS = 5_000;

const defaultSpawner: GeminiProcessSpawner = {
  spawn: (command, args, options) =>
    spawn(command, [...args], {
      cwd: options.cwd,
      detached: options.detached,
      env: options.env,
      stdio: options.stdio,
      windowsHide: options.windowsHide,
    }),
};

export const defaultGeminiCliRunDependencies: GeminiCliRunDependencies = {
  signalProcess: process,
  spawner: defaultSpawner,
  setTimeout,
  clearTimeout,
};

/**
 * Builds the exact headless Gemini CLI invocation. Model identifiers deliberately
 * remain opaque strings: Traycer must pass through whatever the installed CLI
 * exposes instead of carrying a stale hard-coded Gemini model catalogue.
 */
export function buildGeminiCliArgs(
  invocation: GeminiCliInvocation,
): ReadonlyArray<string> {
  const args: string[] = [
    "--prompt",
    invocation.prompt,
    "--output-format",
    "stream-json",
    "--approval-mode",
    invocation.approvalMode,
  ];
  if (invocation.model !== null) {
    args.push("--model", invocation.model);
  }
  if (invocation.resume !== null) {
    args.push("--resume", invocation.resume);
  }
  if (invocation.sessionId !== null) {
    args.push("--session-id", invocation.sessionId);
  }
  if (invocation.skipTrust) {
    args.push("--skip-trust");
  }
  return args;
}

/**
 * Parses one line from Gemini's `--output-format stream-json` output without
 * freezing the parser to the current event catalogue. Unknown event types are
 * preserved as normal events so a future Gemini CLI can evolve independently.
 */
export function parseGeminiStreamLine(rawLine: string): GeminiStreamItem {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawLine);
  } catch (error) {
    return {
      kind: "malformed",
      rawLine,
      reason:
        error instanceof Error ? error.message : "line was not valid JSON",
    };
  }
  if (!isRecord(parsed)) {
    return {
      kind: "malformed",
      rawLine,
      reason: "JSON line was not an object",
    };
  }
  if (typeof parsed.type !== "string" || parsed.type.length === 0) {
    return {
      kind: "malformed",
      rawLine,
      reason: "JSON object did not contain a non-empty string type",
    };
  }
  return {
    kind: "event",
    type: parsed.type,
    raw: parsed,
    rawLine,
  };
}

/**
 * Launches one headless Gemini session, streams every JSONL event to the
 * caller, and terminates the complete child process group on Ctrl-C/TERM.
 * It intentionally never passes Gemini's `--worktree` flag: Traycer owns the
 * selected working directory and Gemini must not create a nested worktree.
 */
export function runGeminiCli(
  invocation: GeminiCliInvocation,
  hooks: GeminiCliRunHooks,
  dependencies: GeminiCliRunDependencies,
): Promise<GeminiCliRunOutcome> {
  return new Promise((resolve, reject) => {
    const child = dependencies.spawner.spawn(
      invocation.binary,
      buildGeminiCliArgs(invocation),
      {
        cwd: invocation.workingDirectory,
        detached: dependencies.signalProcess.platform !== "win32",
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"],
        windowsHide: true,
      },
    );
    const decoder = new StringDecoder("utf8");
    let stdoutBuffer = "";
    let stdoutEnded = false;
    let settled = false;
    let cancellationSignal: NodeJS.Signals | null = null;
    let cancellationKillTimer: NodeJS.Timeout | null = null;
    let sessionId: string | null = null;
    let model: string | null = invocation.model;
    let events = 0;
    let malformedLines = 0;
    let resultStatus: string | null = null;
    let resultError: string | null = null;

    const clearCancellationTimer = (): void => {
      if (cancellationKillTimer !== null) {
        dependencies.clearTimeout(cancellationKillTimer);
        cancellationKillTimer = null;
      }
    };

    const detachSignalHandlers = (): void => {
      dependencies.signalProcess.off("SIGINT", onSigint);
      dependencies.signalProcess.off("SIGTERM", onSigterm);
    };

    const finishStdout = (): void => {
      if (stdoutEnded) return;
      stdoutEnded = true;
      stdoutBuffer += decoder.end();
      if (stdoutBuffer.length > 0) {
        consumeRawLine(stdoutBuffer);
        stdoutBuffer = "";
      }
    };

    const consumeRawLine = (rawLine: string): void => {
      const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
      if (line.length === 0) return;
      const item = parseGeminiStreamLine(line);
      if (item.kind === "malformed") {
        malformedLines += 1;
        hooks.onMalformedLine(item);
        return;
      }
      events += 1;
      if (item.type === "init") {
        const streamedSessionId = readNonEmptyString(item.raw.session_id);
        const streamedModel = readNonEmptyString(item.raw.model);
        if (streamedSessionId !== null) sessionId = streamedSessionId;
        if (streamedModel !== null) model = streamedModel;
      }
      if (item.type === "result") {
        resultStatus = readNonEmptyString(item.raw.status);
        resultError = readResultError(item.raw);
      }
      hooks.onEvent(item);
    };

    const consumeStdoutChunk = (chunk: Buffer): void => {
      stdoutBuffer += decoder.write(chunk);
      for (;;) {
        const newlineIndex = stdoutBuffer.indexOf("\n");
        if (newlineIndex === -1) return;
        const rawLine = stdoutBuffer.slice(0, newlineIndex);
        stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);
        consumeRawLine(rawLine);
      }
    };

    const terminateChild = (signal: NodeJS.Signals): void => {
      if (
        child.pid !== undefined &&
        dependencies.signalProcess.platform !== "win32"
      ) {
        try {
          dependencies.signalProcess.kill(-child.pid, signal);
          return;
        } catch {
          // A process can exit between `pid` observation and group signalling.
          // Fall through to the direct child to keep cancellation idempotent.
        }
      }
      try {
        child.kill(signal);
      } catch {
        // Cancellation is best effort after the child has already exited.
      }
    };

    const requestCancellation = (signal: NodeJS.Signals): void => {
      if (cancellationSignal !== null || settled) return;
      cancellationSignal = signal;
      hooks.onCancellationRequested(signal);
      terminateChild("SIGTERM");
      cancellationKillTimer = dependencies.setTimeout(() => {
        terminateChild("SIGKILL");
      }, CANCELLATION_KILL_GRACE_MS);
    };

    const onSigint = (): void => requestCancellation("SIGINT");
    const onSigterm = (): void => requestCancellation("SIGTERM");

    const settle = (outcome: GeminiCliRunOutcome): void => {
      if (settled) return;
      settled = true;
      clearCancellationTimer();
      detachSignalHandlers();
      resolve(outcome);
    };

    child.stdout?.on("data", consumeStdoutChunk);
    child.stdout?.once("end", finishStdout);
    child.stderr?.on("data", (chunk: Buffer) => {
      hooks.onStderr(chunk.toString("utf8"));
    });
    child.once("error", (error) => {
      if (settled) return;
      clearCancellationTimer();
      detachSignalHandlers();
      settled = true;
      reject(error);
    });
    child.once("close", (code, signal) => {
      finishStdout();
      if (cancellationSignal !== null) {
        settle({
          status: "cancelled",
          exitCode: cancellationSignal === "SIGINT" ? 130 : 143,
          signal: cancellationSignal,
          sessionId,
          model,
          events,
          malformedLines,
          resultError,
        });
        return;
      }
      const succeeded =
        code === 0 && resultStatus === "success" && malformedLines === 0;
      settle({
        status: succeeded ? "success" : "error",
        exitCode: succeeded ? 0 : code === null || code === 0 ? 1 : code,
        signal,
        sessionId,
        model,
        events,
        malformedLines,
        resultError,
      });
    });
    dependencies.signalProcess.on("SIGINT", onSigint);
    dependencies.signalProcess.on("SIGTERM", onSigterm);
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readNonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function readResultError(
  value: Readonly<Record<string, unknown>>,
): string | null {
  if (!isRecord(value.error)) return null;
  return readNonEmptyString(value.error.message);
}
