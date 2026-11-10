export type EngineErrorCode = 'disposed' | 'busy' | 'aborted' | 'engine' | 'worker' | 'protocol' | 'timeout' | 'unsupported_gpu' | 'not_loaded' | 'backpressure';
export class EngineError extends Error { constructor(code: EngineErrorCode, message: string); code: EngineErrorCode }
export interface Adapter { ok: boolean; reason?: string; vendor?: string; architecture?: string; description?: string; fallback?: boolean; maxBufferSize?: number }
export type EngineEvent = { type: 'token'; text: string } | { type: 'prefill'; [key: string]: unknown } | { type: 'tool' | 'tool-result' | 'round'; [key:string]:unknown } | { type: 'done'; reason: string; tokens: number; ms: number; [key: string]: unknown };
export interface ProgressEvent { type: string; [key: string]: unknown }
export interface Engine {
  initialize(options?: { gpu?: 'high-performance' | 'low-power'; signal?: AbortSignal }): Promise<Adapter>;
  load(model: string | { files: File[] }, options?: { contextSize?: number; gpu?: 'high-performance' | 'low-power'; gpuMemoryMB?: number | 'auto'; signal?: AbortSignal; onProgress?: (event: ProgressEvent) => void }): Promise<ProgressEvent>;
  generate(messages: { role: 'system' | 'user' | 'assistant'; content: string }[], options?: { signal?: AbortSignal; maxTokens?: number; temperature?: number; seed?: number; thinking?: boolean; internet?: { enabled: boolean; jinaKey?: string } }): AsyncIterableIterator<EngineEvent>;
  reset(): void;
  dispose(): void;
}
export function createEngine(options?: { workerURL?: string | URL; workerFactory?: (url: string | URL) => Worker; timeoutMs?: number }): Engine;
