/**
 * The assistant: providers, keys and their daily budgets, prompts, the context the model
 * is shown, and the chat. Bodies are snake_case only. Keys never come back — only `hint`.
 */


import { fmt } from '@/lib/format'
import { ApiError, http, qs } from './http'
import type { LLMUsage, Scope, ScopeBody } from './types'

export interface LLMModel {
  id: string
  label: string
  kind: 'text' | 'embedding' | 'open'
  rpm: number
  tpm: number
  rpd: number
  summary: string
  /** Room for volume work. */
  bulk: boolean
  recommended: boolean
  /** Limits are guessed, not published. */
  discovered: boolean
  provider: string
}

export interface LLMModels {
  models: LLMModel[]
  defaults: { chat: string; deep: string }
  note: string
}

export interface LLMProvider {
  id: string
  label: string
  baseUrl: string
  onboardingUrl: string
  keyHint: string
  freeNote: string
  openaiCompatible: boolean
  /** Empty for google: its models are in the roster. */
  models: LLMModel[]
}

export interface LLMProvidersResponse {
  providers: LLMProvider[]
  default: string
  note: string
}

export interface LLMKeyUsage {
  keyId: number
  model: string
  /** Pacific day, YYYY-MM-DD. */
  day: string
  requests: number
  tokens: number
  lastRequestAt: string | null
}

export interface LLMKey {
  id: number
  label: string
  provider: string
  hint: string
  enabled: boolean
  lastOkAt: string | null
  lastError: string | null
  createdAt: string | null
  usage: LLMKeyUsage[]
}

export interface LLMBudget {
  model: string
  label: string
  provider: string
  perKeyPerDay: number
  remainingToday: number
  bulk: boolean
}

export interface LLMKeyStatus {
  keys: LLMKey[]
  enabled: number
  budget: LLMBudget[]
  resetInSeconds: number
  quotaTimezone: string
}

export interface AddKeyRequest {
  key: string
  label?: string | null
  provider?: string
}

export type KeyCheck = { keyId: number; ok: true; models: number; newModels: string[] } | { keyId: number; ok: false; error: string }

export interface PromptInfo {
  slug: string
  label: string
  purpose: string
  context: 'none' | 'catalog_tree' | 'dataset_fields'
  model: string | null
  temperature: number
  body: string
  characters: number
  estimatedTokens: number
}

export interface LLMContextRendered {
  text: string
  scope: string
  counts: { fields: number; datasets: number; categories: number; subcategories: number }
  characters: number
  estimatedTokens: number
}

export type Reasoning = 'quick' | 'normal' | 'careful' | 'deep'

export interface ChatOptions {
  models: LLMModels
  reasoning: { value: Reasoning; level: 'MINIMAL' | 'LOW' | 'MEDIUM' | 'HIGH'; label: string; description: string }[]
  defaultReasoning: Reasoning
  note: string
}

export interface ChatThreadSummary {
  id: number
  title: string
  /** 'USA/D1/TOP3000' */
  scope: string
  updatedAt: string | null
}

export interface ChatPick {
  field: string
  why: string
  [extra: string]: unknown
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  text: string
  meta: { picks?: ChatPick[]; model?: string; reasoning?: Reasoning; tokens?: number }
  createdAt: string | null
}

export interface ChatThread {
  id: number
  title: string
  scope: Scope
  messages: ChatMessage[]
}

export interface ChatSayRequest {
  text: string
  scope: ScopeBody
  thread_id?: number | null
  model?: string | null
  reasoning?: Reasoning
  dataset_ids?: string[]
}

export interface ChatReply {
  threadId: number
  reply: string
  picks: ChatPick[]
  datasets: string[]
  /** Field ids the model named that do not exist in the catalogue. */
  dropped: string[]
  catalogNote: string | null
  usage: LLMUsage
  model: string
  reasoning: Reasoning
}

/** A downloaded catalog scope. Raw row, snake_case. */
export interface DownloadedScope {
  instrument_type: string
  region: string
  delay: number
  universe: string
  fields: number
}

export const llm = {
  providers: () => http.get<LLMProvidersResponse>('/api/llm/providers'),
  models: () => http.get<LLMModels>('/api/llm/models'),
  keys: () => http.get<LLMKeyStatus>('/api/llm/keys'),
  /** 400 llm_error for a duplicate key. */
  addKey: (body: AddKeyRequest) => http.post<LLMKey>('/api/llm/keys', body),
  setEnabled: (id: number, enabled: boolean) => http.put<LLMKey>(`/api/llm/keys/${id}`, { enabled }),
  removeKey: (id: number) => http.del<void>(`/api/llm/keys/${id}`),
  checkKey: (id: number) => http.post<KeyCheck>(`/api/llm/keys/${id}/check`),
  checkAll: () => http.post<KeyCheck[]>('/api/llm/keys/check'),
  prompts: () => http.get<{ prompts: PromptInfo[] }>('/api/llm/prompts'),
  context: (scope: Scope) =>
    http.get<LLMContextRendered>(
      `/api/llm/context${qs({ region: scope.region, delay: scope.delay, universe: scope.universe, instrument_type: scope.instrumentType, rendered: true })}`,
    ),
}

export const chat = {
  options: () => http.get<ChatOptions>('/api/chat/options'),
  threads: (limit = 30) => http.get<ChatThreadSummary[]>(`/api/chat/threads${qs({ limit })}`),
  /** 404 no_such_thread. */
  thread: (id: number) => http.get<ChatThread>(`/api/chat/threads/${id}`),
  deleteThread: (id: number) => http.del<void>(`/api/chat/threads/${id}`),
  /** Spends one assistant request. An undownloaded scope fails with a plain 500. */
  say: (body: ChatSayRequest) => http.post<ChatReply>('/api/chat', body),
  downloadedScopes: () => http.get<DownloadedScope[]>('/api/catalog/scopes'),
}

/** Today's day in the quota's timezone, as usage rows spell it. */
export const quotaDay = (timeZone: string) => new Date().toLocaleDateString('en-CA', { timeZone })

/** The extra facts a 429 llm_budget_exhausted carries: when to retry, and what is left. */
export function budgetDetail(error: unknown): string | null {
  if (!(error instanceof ApiError) || error.code !== 'llm_budget_exhausted') return null
  const keys = (error.body.keys ?? []) as { dailyRemaining?: number }[]
  const left = keys.reduce((sum, k) => sum + (k.dailyRemaining ?? 0), 0)
  return `Retry in ${fmt.duration(error.body.retryAfter)} · ${fmt.int(left)} requests left today across ${fmt.int(keys.length)} key budgets.`
}
