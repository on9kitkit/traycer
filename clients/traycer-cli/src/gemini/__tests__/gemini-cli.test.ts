import { EventEmitter } from "node:events";
import { PassThrough } from "node:stream";
import { describe, expect, it } from "vitest";
import {
  buildGeminiCliArgs,
  parseGeminiStreamLine,
  runGeminiCli,
  type GeminiCliInvocation,
  type GeminiCliRunDependencies,
  type GeminiCliRunHooks,
  type GeminiMalformedStreamLine,
  type GeminiProcess,
  type GeminiStreamEvent,
} from "../gemini-cli";

class FakeGeminiProcess extends EventEmitter implements GeminiProcess {
  readonly pid = 4242;
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  readonly directKills: NodeJS.Signals[] = [];

  kill(signal: NodeJS.Signals | undefined): boolean {
    this.directKills.push(signal ?? "SIGTERM");
    return true;
  }

  close(code: number | null, signal: NodeJS.Signals | null): void {
    this.emit("close", code, signal);
  }
}

class FakeSignalProcess extends EventEmitter {
  readonly platform: NodeJS.Platform = "darwin";
  readonly processGroupKills: Array<{
    readonly pid: number;
    readonly signal: NodeJS.Signals;
  }> = [];

  kill(pid: number, signal: NodeJS.Signals): void {
    this.processGroupKills.push({ pid, signal });
  }
}

const invocation: GeminiCliInvocation = {
  approvalMode: "auto_edit",
  binary: "/opt/homebrew/bin/gemini",
  model: "gemini-future-preview",
  prompt: "Fix the failing test and verify it.",
  resume: null,
  sessionId: null,
  skipTrust: true,
  workingDirectory: "/tmp/linked-worktree",
};

function captureHooks(): {
  readonly hooks: GeminiCliRunHooks;
  readonly events: GeminiStreamEvent[];
  readonly malformed: GeminiMalformedStreamLine[];
  readonly stderr: string[];
  readonly cancellations: NodeJS.Signals[];
} {
  const events: GeminiStreamEvent[] = [];
  const malformed: GeminiMalformedStreamLine[] = [];
  const stderr: string[] = [];
  const cancellations: NodeJS.Signals[] = [];
  return {
    hooks: {
      onCancellationRequested: (signal) => cancellations.push(signal),
      onEvent: (event) => events.push(event),
      onMalformedLine: (line) => malformed.push(line),
      onStderr: (text) => stderr.push(text),
    },
    events,
    malformed,
    stderr,
    cancellations,
  };
}

function createDependencies(
  child: FakeGeminiProcess,
  signalProcess: FakeSignalProcess,
): {
  readonly dependencies: GeminiCliRunDependencies;
  readonly scheduledCancellationEscalations: Array<() => void>;
  readonly spawnCalls: Array<{
    readonly command: string;
    readonly args: readonly string[];
    readonly cwd: string;
    readonly detached: boolean;
  }>;
} {
  const scheduledCancellationEscalations: Array<() => void> = [];
  const spawnCalls: Array<{
    readonly command: string;
    readonly args: readonly string[];
    readonly cwd: string;
    readonly detached: boolean;
  }> = [];
  const dependencies: GeminiCliRunDependencies = {
    clearTimeout: (timeout) => globalThis.clearTimeout(timeout),
    setTimeout: (callback) => {
      scheduledCancellationEscalations.push(callback);
      return globalThis.setTimeout(() => undefined, 60_000);
    },
    signalProcess,
    spawner: {
      spawn: (command, args, options) => {
        spawnCalls.push({
          args,
          command,
          cwd: options.cwd,
          detached: options.detached,
        });
        return child;
      },
    },
  };
  return { dependencies, scheduledCancellationEscalations, spawnCalls };
}

describe("buildGeminiCliArgs", () => {
  it("uses Gemini headless JSONL mode and passes an unknown future model through unchanged", () => {
    expect(buildGeminiCliArgs(invocation)).toEqual([
      "--prompt",
      "Fix the failing test and verify it.",
      "--output-format",
      "stream-json",
      "--approval-mode",
      "auto_edit",
      "--model",
      "gemini-future-preview",
      "--skip-trust",
    ]);
  });

  it("keeps resume and supplied session identifiers as separate optional CLI flags", () => {
    expect(
      buildGeminiCliArgs({
        ...invocation,
        resume: "session-to-resume",
        sessionId: null,
        skipTrust: false,
      }),
    ).toContain("--resume");
    expect(
      buildGeminiCliArgs({
        ...invocation,
        resume: null,
        sessionId: "session-to-start",
      }),
    ).toContain("--session-id");
  });
});

describe("parseGeminiStreamLine", () => {
  it("preserves unknown event types for a newer Gemini CLI", () => {
    expect(
      parseGeminiStreamLine(
        '{"type":"future_agent_event","payload":{"value":1}}',
      ),
    ).toMatchObject({
      kind: "event",
      type: "future_agent_event",
    });
  });

  it("quarantines malformed stream output rather than throwing", () => {
    expect(parseGeminiStreamLine("not json")).toMatchObject({
      kind: "malformed",
      rawLine: "not json",
    });
    expect(parseGeminiStreamLine('{"message":"missing type"}')).toMatchObject({
      kind: "malformed",
      reason: "JSON object did not contain a non-empty string type",
    });
  });
});

describe("runGeminiCli", () => {
  it("streams chunked JSONL, records session metadata, and keeps unknown events", async () => {
    const child = new FakeGeminiProcess();
    const signalProcess = new FakeSignalProcess();
    const capture = captureHooks();
    const { dependencies, spawnCalls } = createDependencies(
      child,
      signalProcess,
    );

    const completion = runGeminiCli(invocation, capture.hooks, dependencies);
    child.stdout.write(
      '{"type":"init","session_id":"session-1","model":"gemini-',
    );
    child.stdout.write(
      'future-preview"}\n{"type":"message","role":"assistant","content":"Working"}\n',
    );
    child.stdout.write('{"type":"future_agent_event","payload":true}\n');
    child.stdout.write('{"type":"result","status":"success"}');
    child.close(0, null);

    await expect(completion).resolves.toEqual({
      events: 4,
      exitCode: 0,
      malformedLines: 0,
      model: "gemini-future-preview",
      resultError: null,
      sessionId: "session-1",
      signal: null,
      status: "success",
    });
    expect(capture.events.map((event) => event.type)).toEqual([
      "init",
      "message",
      "future_agent_event",
      "result",
    ]);
    expect(spawnCalls).toEqual([
      {
        args: buildGeminiCliArgs(invocation),
        command: "/opt/homebrew/bin/gemini",
        cwd: "/tmp/linked-worktree",
        detached: true,
      },
    ]);
  });

  it("treats malformed stdout as a failed run even if Gemini exits zero", async () => {
    const child = new FakeGeminiProcess();
    const capture = captureHooks();
    const { dependencies } = createDependencies(child, new FakeSignalProcess());

    const completion = runGeminiCli(invocation, capture.hooks, dependencies);
    child.stdout.write("not JSON\n");
    child.stdout.write('{"type":"result","status":"success"}\n');
    child.close(0, null);

    await expect(completion).resolves.toMatchObject({
      exitCode: 1,
      malformedLines: 1,
      status: "error",
    });
    expect(capture.malformed).toHaveLength(1);
  });

  it("surfaces a terminal Gemini API error even if the process exits zero", async () => {
    const child = new FakeGeminiProcess();
    const capture = captureHooks();
    const { dependencies } = createDependencies(child, new FakeSignalProcess());

    const completion = runGeminiCli(invocation, capture.hooks, dependencies);
    child.stdout.write(
      '{"type":"result","status":"error","error":{"type":"unknown","message":"An unknown API error occurred."}}\n',
    );
    child.close(0, null);

    await expect(completion).resolves.toMatchObject({
      exitCode: 1,
      resultError: "An unknown API error occurred.",
      status: "error",
    });
  });

  it("cancels the full POSIX process group and removes its signal listeners", async () => {
    const child = new FakeGeminiProcess();
    const signalProcess = new FakeSignalProcess();
    const capture = captureHooks();
    const { dependencies, scheduledCancellationEscalations } =
      createDependencies(child, signalProcess);

    const completion = runGeminiCli(invocation, capture.hooks, dependencies);
    signalProcess.emit("SIGINT");
    child.close(null, "SIGTERM");

    await expect(completion).resolves.toMatchObject({
      exitCode: 130,
      signal: "SIGINT",
      status: "cancelled",
    });
    expect(capture.cancellations).toEqual(["SIGINT"]);
    expect(signalProcess.processGroupKills).toEqual([
      { pid: -4242, signal: "SIGTERM" },
    ]);
    expect(scheduledCancellationEscalations).toHaveLength(1);
    expect(signalProcess.listenerCount("SIGINT")).toBe(0);
    expect(signalProcess.listenerCount("SIGTERM")).toBe(0);
  });
});
