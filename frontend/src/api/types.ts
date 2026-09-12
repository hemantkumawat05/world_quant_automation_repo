/**
 * Types shared by more than one screen. Screen-specific contracts live next to their
 * endpoints in `lib/api/<domain>.ts`.
 *
 * Units: turnover, returns, drawdown, margin, coverage and truncation are FRACTIONS
 * (0.64 = 64%). Sharpe and Fitness are plain ratios. Timestamps are ISO strings; one
 * without an offset is UTC.
 *
 * Casing on the wire is not uniform — see frontend/CLAUDE.md "API casing".
 */

/** The four-part address almost every lab and catalog call takes. camelCase in the UI. */
export interface Scope {
  instrumentType: string
  region: string
  delay: number
  universe: string
}

/** The same scope for routes whose bodies accept snake_case only. */
export interface ScopeBody {
  instrument_type: string
  region: string
  delay: number
  universe: string
}

export const toScopeBody = (scope: Scope): ScopeBody => ({
  instrument_type: scope.instrumentType,
  region: scope.region,
  delay: scope.delay,
  universe: scope.universe,
})

export const scopeLabel = (scope: Scope): string => `${scope.region} · D${scope.delay} · ${scope.universe}`

/** BRAIN simulation settings, as BRAIN names them. */
export interface SimulationSettings {
  instrumentType?: string
  region: string
  universe: string
  delay: number
  decay?: number
  neutralization?: string
  truncation?: number
  pasteurization?: string
  unitHandling?: string
  nanHandling?: string
  language?: string
  visualization?: boolean
  testPeriod?: string | null
  maxTrade?: string | null
  maxPosition?: string | null
  [extra: string]: unknown
}

export type SimStatus =
  | 'QUEUED'
  | 'PENDING'
  | 'RUNNING'
  | 'COMPLETE'
  | 'WARNING'
  | 'ERROR'
  | 'FAILED'
  | 'CANCELLED'
  | 'TIMEOUT'
  | 'ORPHANED'
  | 'REJECTED'
  | 'SKIPPED'

/** One simulation record. A multi-simulation parent holds one of the 8 slots. */
export interface SimulationRow {
  id: number
  platformId: string | null
  alphaId: string | null
  status: SimStatus
  platformStatus: 'COMPLETE' | 'WARNING' | 'ERROR' | 'FAIL' | 'CANCELLED' | 'TIMEOUT' | null
  /** 0..1, reported on batch parents and standalone simulations only. */
  progress: number | null
  message: string | null
  /** A batch parent reads "<n> simulations". */
  expression: string | null
  task: string
  region: string
  delay: number
  universe: string | null
  instrumentType: string
  language: string
  simType: string
  isBatch: boolean
  childIds: string[]
  /** The batch parent's record id; null for standalone simulations and for batch parents. */
  parentId: number | null
  createdAt: string | null
  /** Identical on a batch parent and its children; the only link between them. */
  submittedAt: string | null
  finishedAt: string | null
  elapsedSeconds: number | null
  settings: SimulationSettings | null
}

export interface EngineStatus {
  /** 8 */
  slots: number
  /** 10, or 1 without the MULTI_SIMULATION permission. */
  maxBatch: number
  slotsUsed: number
  slotsFree: number
  /** Task name → queued simulations. */
  queued: Record<string, number>
  queuedTotal: number
  /** Task name → slots held. */
  inFlight: Record<string, number>
  quotas: Record<string, number>
  dailyLimitHit: boolean
}

export interface EnqueueOutcome {
  index: number
  recordId: number
  status: 'QUEUED' | 'SKIPPED'
  alphaId: string | null
  hash: string
}

/** What every call that queues simulations adds to its response. */
export interface EnqueueResult {
  queued: number[]
  skipped: { alphaId: string | null; hash: string }[]
  outcomes?: EnqueueOutcome[]
  task?: string
  status?: EngineStatus
}

export type CheckResult = 'PASS' | 'FAIL' | 'PENDING' | 'WARNING' | 'ERROR'

export interface AlphaCheck {
  name: string
  result: CheckResult | null
  limit?: number | null
  value?: number | null
  message?: string | null
  [extra: string]: unknown
}

export interface LLMUsage {
  promptTokens: number
  outputTokens: number
  thinkingTokens: number
  totalTokens: number
}

/** One market's state inside a whole-catalog sync, for the sync matrix. */
export interface SyncMarket {
  region: string
  delay: number
  universe: string
  state: 'waiting' | 'fetching' | 'fields' | 'details' | 'done' | 'failed'
  fields: number | null
}

export interface SyncRun {
  id: number
  instrumentType: string
  region: string
  delay: number
  universe: string
  label: string
  status: 'RUNNING' | 'COMPLETE' | 'FAILED' | 'CANCELLED'
  phase: 'categories' | 'datasets' | 'fields' | 'details' | null
  cursorOffset: number | null
  cursorDataset: string | null
  categoriesSynced: number
  datasetsSynced: number
  fieldsSynced: number
  fieldsExpected: number | null
  /** 0..1 */
  fraction: number | null
  truncatedDatasets: unknown[]
  error: string | null
  startedAt: string | null
  finishedAt: string | null
  /** A whole-catalog sync (every market) rather than one scope. */
  all?: boolean
  /** Live only, while a whole-catalog sync runs. */
  stage?: 'fields' | 'details'
  scopesDone?: number
  scopesTotal?: number
  markets?: SyncMarket[]
}

export interface Session {
  authenticated: boolean
  userId: string | null
  fullName: string | null
  permissions: string[]
  /** Epoch seconds. */
  expiresAt: number | null
  expiresInSeconds: number | null
  restoredFromCache: boolean
  verificationUrl: string | null
  detail: string | null
  canMultiSimulate: boolean
  isConsultant: boolean
  token?: string | null
}

export interface Today {
  step: 'sign-in' | 'add-key' | 'ready'
  you: {
    signedIn: boolean
    email: string | null
    userId: string | null
    fullName: string | null
    features: { code: string; label: string; meaning: string }[]
    canRunTenAtOnce: boolean
    verificationUrl: string | null
  }
  simulations: {
    limit: number
    used: number
    remaining: number
    /** Sent but not yet counted by BRAIN's lagging header; already taken off `remaining`. */
    pendingCharge: number
    queued: number
    unspoken: number
    /** True once the platform's own figure from today is known. */
    exact: boolean
    resetsInSeconds: number
    resetsAt: string
    engine: EngineStatus
    headline: string
  }
  assistant: {
    keys: number
    enabledKeys: number
    requestsRemainingToday: number
    budget: { model: string; label: string; provider: string; perKeyPerDay: number; remainingToday: number; bulk: boolean }[]
    resetsInSeconds: number
    resetsAt: string
    headline: string
  }
  catalog: { scope: string; synced: boolean; fields: number; running: SyncRun | null; anySynced: boolean }
}

export interface BarStatus {
  signedIn: boolean
  fullName: string | null
  expiresInSeconds: number | null
  simulations: { remaining: number; limit: number; exact: boolean; queued: number }
  resetsInSeconds: number
}

/** One US Eastern day of BRAIN's own activity record. */
export interface ActivityDay {
  date: string
  simulations: number
  submissions: number
}

export interface BackgroundTask {
  id: string
  kind: string
  label: string
  /** 0..1 */
  progress: number | null
  detail: string
  state: 'running' | 'done' | 'failed' | 'cancelled'
  error: string | null
  elapsedSeconds: number
  meta: Record<string, unknown>
}

export interface TasksSummary {
  busy: boolean
  running: number
  failed: number
  progress: number | null
  tasks: BackgroundTask[]
}
