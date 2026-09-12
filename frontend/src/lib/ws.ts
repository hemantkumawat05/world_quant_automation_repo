/**
 * Live telemetry over one WebSocket.
 *
 * The backend pushes on a single multiplexed channel and replays each topic's last
 * message on connect. Reconnects with backoff, because a local backend restarts often.
 *
 * Topics (see backend `realtime.py`):
 * - `simulations` — the full active list (`SimulationRow[]`) on every change; ALSO
 *   `{alphaId, submittable}` when a backfill resolves checks, and `{alphaId, stored}` once a
 *   finished simulation's Alpha is saved locally. Guard with `Array.isArray`.
 * - `tasks` — `TasksSummary` on any background task change.
 * - `sync` — `SyncRun` while a catalog download progresses.
 * - `studies` — `{kind:"studies"}`, a signal to refetch studies, Template Lab and GA runs.
 * - `session` — `Session` on sign-in, sign-out and renewal.
 */

import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

export type Topic = 'simulations' | 'tasks' | 'sync' | 'studies' | 'session'

type Handler = (payload: unknown) => void

const RECONNECT_MIN = 500
const RECONNECT_MAX = 10_000

class Telemetry {
  private socket: WebSocket | null = null
  private handlers = new Map<string, Set<Handler>>()
  private statusHandlers = new Set<(connected: boolean) => void>()
  private delay = RECONNECT_MIN
  private connected = false

  reconnect(): void {
    if (this.socket) {
      this.socket.onclose = null
      this.socket.onerror = null
      this.socket.close()
      this.socket = null
    }
    this.delay = RECONNECT_MIN
    this.connect()
  }

  connect(): void {
    if (this.socket && this.socket.readyState <= WebSocket.OPEN) return
    let wsUrl = import.meta.env.VITE_WS_URL
    if (!wsUrl) {
      const apiUrl = import.meta.env.VITE_API_URL
      if (apiUrl) {
        try {
          const parsed = new URL(apiUrl)
          const wsProtocol = parsed.protocol === 'https:' ? 'wss:' : 'ws:'
          wsUrl = `${wsProtocol}//${parsed.host}/ws`
        } catch {
          const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
          wsUrl = `${protocol}//${location.host}/ws`
        }
      } else {
        const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
        wsUrl = `${protocol}//${location.host}/ws`
      }
    }

    const token = typeof window !== 'undefined' ? localStorage.getItem('alpha_token') : null
    if (token) {
      const separator = wsUrl.includes('?') ? '&' : '?'
      wsUrl = `${wsUrl}${separator}token=${encodeURIComponent(token)}`
    }

    const socket = new WebSocket(wsUrl)
    this.socket = socket

    socket.onopen = () => {
      this.delay = RECONNECT_MIN
      this.setConnected(true)
    }
    socket.onmessage = (event) => {
      let envelope: { topic?: string; payload?: unknown }
      try {
        envelope = JSON.parse(event.data)
      } catch {
        return
      }
      if (!envelope.topic || envelope.topic === 'ping') return
      this.handlers.get(envelope.topic)?.forEach((handler) => handler(envelope.payload))
    }
    socket.onclose = () => {
      this.setConnected(false)
      this.socket = null
      setTimeout(() => this.connect(), this.delay)
      this.delay = Math.min(this.delay * 2, RECONNECT_MAX)
    }
    socket.onerror = () => socket.close()
  }

  private setConnected(connected: boolean): void {
    this.connected = connected
    this.statusHandlers.forEach((handler) => handler(connected))
  }

  subscribe(topic: Topic, handler: Handler): () => void {
    const set = this.handlers.get(topic) ?? new Set()
    set.add(handler)
    this.handlers.set(topic, set)
    return () => set.delete(handler)
  }

  onStatus(handler: (connected: boolean) => void): () => void {
    this.statusHandlers.add(handler)
    // Report the current state at once: the socket opens long before components mount.
    handler(this.connected)
    return () => this.statusHandlers.delete(handler)
  }
}

export const telemetry = new Telemetry()

/** Run `handler` for every message on `topic`. The handler may change freely. */
export function useTopic(topic: Topic, handler: (payload: unknown) => void): void {
  const ref = useRef(handler)
  useEffect(() => {
    ref.current = handler
  })
  useEffect(() => telemetry.subscribe(topic, (payload) => ref.current(payload)), [topic])
}

/** Whether the telemetry socket is connected. Poll only while it is not. */
export function useConnected(): boolean {
  const [connected, setConnected] = useState(false)
  useEffect(() => telemetry.onStatus(setConnected), [])
  return connected
}

/**
 * Refetch queries under `queryKey` whenever `topic` reports a change.
 *
 * The socket is used as a signal, not as state: a drifting payload shape can never
 * desync the UI. Bursts collapse to at most one refetch per `minGapMs`, with a trailing
 * call so the final state still lands.
 */
export function useRefetchOn(topic: Topic, queryKey: readonly unknown[], minGapMs = 1000): void {
  const queryClient = useQueryClient()
  const last = useRef(0)
  const pending = useRef<ReturnType<typeof setTimeout> | null>(null)
  const keyRef = useRef(queryKey)
  useEffect(() => {
    keyRef.current = queryKey
  })

  useTopic(topic, () => {
    const fire = () => {
      last.current = Date.now()
      pending.current = null
      void queryClient.invalidateQueries({ queryKey: keyRef.current })
    }
    const since = Date.now() - last.current
    if (since >= minGapMs) fire()
    else if (!pending.current) pending.current = setTimeout(fire, minGapMs - since)
  })

  useEffect(
    () => () => {
      if (pending.current) clearTimeout(pending.current)
    },
    [],
  )
}
