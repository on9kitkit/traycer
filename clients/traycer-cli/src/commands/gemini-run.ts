import { constants } from "node:fs";
import { access, lstat, open, realpath } from "node:fs/promises";
import { isAbsolute, join } from "node:path";
import {
  GEMINI_APPROVAL_MODES,
  defaultGeminiCliRunDependencies,
  runGeminiCli,
  type GeminiApprovalMode,
  type GeminiCliRunOutcome,
  type GeminiStreamEvent,
} from "../gemini/gemini-cli";
import { ProcessRunError, runCommand } from "../service/process-runner";
import { CLI_ERROR_CODES, cliError } from "../runner/errors";
import type { CommandFn } from "../runner/runner";
import { writeStderr, writeStdout } from "../runner/std-write";

const GEMINI_VERSION_TIMEOUT_MS = 5_000;
const GIT_PREFLIGHT_TIMEOUT_MS = 5_000;

export interface GeminiRunCommandOptions {
  readonly allowPrimaryWorktree: boolean;
  readonly binary: string | null;
  readonly cwd: string;
  readonly json: boolean;
  readonly model: string | null;
  readonly permissionMode: string | null;
  readonly prompt: string;
  readonly resume: string | null;
  readonly sessionId: string | null;
  readonly skipTrust: boolean;
  readonly transcriptPath: string | null;
}

export interface GeminiCliInspection {
  readonly auth: "not_probed";
  readonly available: boolean;
  readonly binary: string;
  readonly error: string | null;
  readonly modelSelection: "opaque_cli_passthrough";
  readonly version: string | null;
}

export interface GeminiRunCommandResult extends GeminiCliRunOutcome {
  readonly binary: string;
  readonly transcriptPath: string | null;
  readonly workingDirectory: string;
}

/**
 * Returns installation evidence without initiating an OAuth flow. Gemini CLI
 * authentication is interactive/provider-owned, so this deliberately reports
 * it as not probed rather than claiming that an executable implies a usable
 * account.
 */
export async function inspectGeminiCli(
  binary: string,
): Promise<GeminiCliInspection> {
  try {
    const result = await runCommand(binary, ["--version"], {
      cwd: undefined,
      env: undefined,
      timeoutMs: GEMINI_VERSION_TIMEOUT_MS,
      tolerateNonZeroExit: false,
    });
    const version = result.stdout.trim() || null;
    return {
      auth: "not_probed",
      available: true,
      binary,
      error: null,
      modelSelection: "opaque_cli_passthrough",
      version,
    };
  } catch (error) {
    return {
      auth: "not_probed",
      available: false,
      binary,
      error: describeProcessFailure(error),
      modelSelection: "opaque_cli_passthrough",
      version: null,
    };
  }
}

export function buildGeminiStatusCommand(opts: {
  readonly binary: string | null;
}): CommandFn {
  return async () => {
    const binary = resolveGeminiBinary(opts.binary);
    const inspection = await inspectGeminiCli(binary);
    return {
      data: inspection,
      human: formatGeminiInspection(inspection),
      exitCode: inspection.available ? 0 : 1,
    };
  };
}

/**
 * Runs Gemini only after proving the selected directory is the root of a Git
 * worktree. A normal checkout is refused by default: callers should create a
 * linked worktree first with `traycer worktree create`, then pass that returned
 * path here. The explicit opt-out is useful only for deliberate one-off runs.
 */
export async function runGeminiRunCommand(
  options: GeminiRunCommandOptions,
): Promise<GeminiRunCommandResult> {
  const binary = resolveGeminiBinary(options.binary);
  const prompt = requireNonEmpty(options.prompt, "--prompt");
  const approvalMode = resolveApprovalMode(options.permissionMode);
  const resume = resolveOptionalValue(options.resume, "--resume");
  const sessionId = resolveOptionalValue(options.sessionId, "--session-id");
  if (resume !== null && sessionId !== null) {
    throw cliError({
      code: CLI_ERROR_CODES.INVALID_ARGUMENT,
      message:
        "traycer gemini run: --resume and --session-id cannot be combined; choose the Gemini session selection mechanism you intend to use.",
      details: null,
      exitCode: 1,
    });
  }
  const model = resolveOptionalValue(options.model, "--model");
  const workingDirectory = await resolveGeminiWorktree(
    options.cwd,
    options.allowPrimaryWorktree,
  );
  const inspection = await inspectGeminiCli(binary);
  if (!inspection.available) {
    throw cliError({
      code: CLI_ERROR_CODES.NOT_FOUND,
      message: `traycer gemini run: could not execute Gemini CLI '${binary}': ${inspection.error ?? "unknown failure"}`,
      details: { binary, error: inspection.error },
      exitCode: 1,
    });
  }

  const transcript = await createTranscript(options.transcriptPath);
  const renderer = options.json ? null : new GeminiHumanRenderer();
  let outcome: GeminiCliRunOutcome;
  try {
    outcome = await runGeminiCli(
      {
        approvalMode,
        binary,
        model,
        prompt,
        resume,
        sessionId,
        skipTrust: options.skipTrust,
        workingDirectory,
      },
      {
        onCancellationRequested: (signal) => {
          renderer?.finishAssistantText();
          writeStderr(`\n[gemini] cancellation requested (${signal})\n`);
        },
        onEvent: (event) => {
          transcript.append(event.rawLine);
          if (options.json) {
            writeStdout(`${event.rawLine}\n`);
            return;
          }
          renderer?.render(event);
        },
        onMalformedLine: (line) => {
          transcript.append(line.rawLine);
          writeStderr(
            `[gemini] ignored malformed stream line: ${line.reason}\n`,
          );
        },
        onStderr: (text) => writeStderr(text),
      },
      defaultGeminiCliRunDependencies,
    );
  } finally {
    renderer?.finishAssistantText();
    await transcript.close();
  }

  if (!options.json) {
    writeGeminiOutcome(outcome, transcript.path);
  }
  return {
    ...outcome,
    binary,
    transcriptPath: transcript.path,
    workingDirectory,
  };
}

export async function resolveGeminiWorktree(
  rawPath: string,
  allowPrimaryWorktree: boolean,
): Promise<string> {
  const trimmed = rawPath.trim();
  if (!isAbsolute(trimmed)) {
    throw cliError({
      code: CLI_ERROR_CODES.INVALID_ARGUMENT,
      message:
        "traycer gemini run: --cwd must be an absolute path to a Git worktree.",
      details: { cwd: rawPath },
      exitCode: 1,
    });
  }
  let workingDirectory: string;
  try {
    workingDirectory = await realpath(trimmed);
  } catch (error) {
    throw cliError({
      code: CLI_ERROR_CODES.NOT_FOUND,
      message: `traycer gemini run: --cwd does not resolve to a directory: ${trimmed}`,
      details: { cwd: trimmed, error: describeProcessFailure(error) },
      exitCode: 1,
    });
  }
  try {
    await access(workingDirectory, constants.R_OK | constants.X_OK);
  } catch (error) {
    throw cliError({
      code: CLI_ERROR_CODES.FORBIDDEN,
      message: `traycer gemini run: cannot access selected worktree ${workingDirectory}.`,
      details: { cwd: workingDirectory, error: describeProcessFailure(error) },
      exitCode: 1,
    });
  }
  let gitMarkerIsFile = false;
  try {
    gitMarkerIsFile = (await lstat(join(workingDirectory, ".git"))).isFile();
  } catch {
    // The Git command below supplies the more useful diagnostic for paths that
    // are not repositories. This is only the linked-worktree isolation check.
  }
  if (!allowPrimaryWorktree && !gitMarkerIsFile) {
    throw cliError({
      code: CLI_ERROR_CODES.INVALID_ARGUMENT,
      message:
        "traycer gemini run: --cwd must be a linked Git worktree created for this run. Create one with `traycer worktree create`, then pass its returned path; use --allow-primary-worktree only when you explicitly accept edits in the primary checkout.",
      details: { cwd: workingDirectory },
      exitCode: 1,
    });
  }
  const worktreeRoot = await readGitWorktreeRoot(workingDirectory);
  if (worktreeRoot !== workingDirectory) {
    throw cliError({
      code: CLI_ERROR_CODES.INVALID_ARGUMENT,
      message:
        "traycer gemini run: --cwd must name the root of the selected Git worktree, not a subdirectory.",
      details: { cwd: workingDirectory, worktreeRoot },
      exitCode: 1,
    });
  }
  return workingDirectory;
}

function resolveGeminiBinary(rawBinary: string | null): string {
  if (rawBinary === null) return "gemini";
  const trimmed = rawBinary.trim();
  if (!isAbsolute(trimmed)) {
    throw cliError({
      code: CLI_ERROR_CODES.INVALID_ARGUMENT,
      message:
        "traycer gemini: --binary must be an absolute executable path; omit it to use `gemini` from PATH.",
      details: { binary: rawBinary },
      exitCode: 1,
    });
  }
  return trimmed;
}

function resolveApprovalMode(rawMode: string | null): GeminiApprovalMode {
  if (rawMode === null) return "default";
  const mode = rawMode.trim();
  for (const knownMode of GEMINI_APPROVAL_MODES) {
    if (knownMode === mode) return knownMode;
  }
  throw cliError({
    code: CLI_ERROR_CODES.INVALID_ARGUMENT,
    message: `traycer gemini run: --permission-mode must be one of ${GEMINI_APPROVAL_MODES.join(", ")}.`,
    details: { permissionMode: rawMode },
    exitCode: 1,
  });
}

function resolveOptionalValue(
  rawValue: string | null,
  flag: string,
): string | null {
  if (rawValue === null) return null;
  const value = rawValue.trim();
  if (value.length === 0) {
    throw cliError({
      code: CLI_ERROR_CODES.INVALID_ARGUMENT,
      message: `traycer gemini run: ${flag} must not be empty when supplied.`,
      details: { flag },
      exitCode: 1,
    });
  }
  return value;
}

function requireNonEmpty(rawValue: string, flag: string): string {
  const value = rawValue.trim();
  if (value.length > 0) return value;
  throw cliError({
    code: CLI_ERROR_CODES.INVALID_ARGUMENT,
    message: `traycer gemini run: ${flag} must not be empty.`,
    details: { flag },
    exitCode: 1,
  });
}

async function readGitWorktreeRoot(workingDirectory: string): Promise<string> {
  try {
    const result = await runCommand(
      "git",
      ["-C", workingDirectory, "rev-parse", "--show-toplevel"],
      {
        cwd: undefined,
        env: undefined,
        timeoutMs: GIT_PREFLIGHT_TIMEOUT_MS,
        tolerateNonZeroExit: false,
      },
    );
    return await realpath(result.stdout.trim());
  } catch (error) {
    throw cliError({
      code: CLI_ERROR_CODES.INVALID_ARGUMENT,
      message: `traycer gemini run: --cwd is not a Git worktree: ${workingDirectory}`,
      details: { cwd: workingDirectory, error: describeProcessFailure(error) },
      exitCode: 1,
    });
  }
}

function formatGeminiInspection(inspection: GeminiCliInspection): string {
  if (!inspection.available) {
    return [
      "Gemini CLI is unavailable.",
      `  Binary: ${inspection.binary}`,
      `  Error: ${inspection.error ?? "unknown failure"}`,
      "  Install or authenticate Gemini CLI separately, then re-run this command.",
    ].join("\n");
  }
  return [
    "Gemini CLI is available.",
    `  Binary: ${inspection.binary}`,
    `  Version: ${inspection.version ?? "not reported"}`,
    "  Authentication: not probed (Gemini CLI owns its interactive login flow)",
    "  Model selection: opaque CLI passthrough (no Traycer model allowlist)",
  ].join("\n");
}

function describeProcessFailure(error: unknown): string {
  if (error instanceof ProcessRunError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}

interface GeminiTranscript {
  readonly path: string | null;
  append(rawLine: string): void;
  close(): Promise<void>;
}

async function createTranscript(
  rawPath: string | null,
): Promise<GeminiTranscript> {
  if (rawPath === null) return noTranscript;
  const path = rawPath.trim();
  if (!isAbsolute(path)) {
    throw cliError({
      code: CLI_ERROR_CODES.INVALID_ARGUMENT,
      message:
        "traycer gemini run: --transcript must be an absolute file path when supplied.",
      details: { transcript: rawPath },
      exitCode: 1,
    });
  }
  const handle = await open(path, "w");
  let writes: Promise<void> = Promise.resolve();
  return {
    path,
    append: (rawLine) => {
      writes = writes.then(async () => {
        await handle.write(`${rawLine}\n`);
      });
    },
    close: async () => {
      await writes;
      await handle.close();
    },
  };
}

const noTranscript: GeminiTranscript = {
  path: null,
  append: () => undefined,
  close: async () => undefined,
};

class GeminiHumanRenderer {
  private assistantTextOpen = false;

  render(event: GeminiStreamEvent): void {
    switch (event.type) {
      case "init": {
        const session = readString(event.raw.session_id);
        const model = readString(event.raw.model);
        const detail = [
          model === null ? null : `model ${model}`,
          session === null ? null : `session ${session}`,
        ]
          .filter((value): value is string => value !== null)
          .join(", ");
        if (detail.length > 0) writeStderr(`[gemini] ${detail}\n`);
        return;
      }
      case "message": {
        if (readString(event.raw.role) !== "assistant") return;
        const content = readString(event.raw.content);
        if (content === null) return;
        writeStdout(content);
        this.assistantTextOpen = true;
        return;
      }
      case "tool_use": {
        this.finishAssistantText();
        const toolName = readString(event.raw.tool_name) ?? "unknown tool";
        writeStderr(`[gemini] tool: ${toolName}\n`);
        return;
      }
      case "tool_result": {
        const status = readString(event.raw.status) ?? "unknown";
        writeStderr(`[gemini] tool result: ${status}\n`);
        return;
      }
      case "error": {
        this.finishAssistantText();
        const message =
          readString(event.raw.message) ?? "Gemini emitted an error";
        writeStderr(`[gemini] ${message}\n`);
        return;
      }
      case "result":
        this.finishAssistantText();
        return;
      default:
        return;
    }
  }

  finishAssistantText(): void {
    if (!this.assistantTextOpen) return;
    writeStdout("\n");
    this.assistantTextOpen = false;
  }
}

function writeGeminiOutcome(
  outcome: GeminiCliRunOutcome,
  transcriptPath: string | null,
): void {
  const transcript =
    transcriptPath === null ? "" : ` Transcript: ${transcriptPath}.`;
  if (outcome.status === "success") {
    writeStderr(
      `[gemini] completed with ${outcome.events} stream events.${transcript}\n`,
    );
    return;
  }
  const detail = outcome.resultError === null ? "" : ` ${outcome.resultError}`;
  writeStderr(
    `[gemini] ${outcome.status} (exit ${outcome.exitCode}).${detail}${transcript}\n`,
  );
}

function readString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}
