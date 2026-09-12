/**
 * The one gate. Nothing works without a BRAIN session.
 */

import { useState, type FormEvent } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { auth } from '@/api/core'
import { ApiError } from '@/api/http'
import { useLive } from '@/lib/live'
import { telemetry } from '@/lib/ws'
import { Button, ErrorNotice, Field, Input } from '@/ui/kit'

/** Larger than the in-app controls: this screen is the whole page and is read from arm's length. */
const FIELD = '[&>span:first-child]:text-[13px]'
const INPUT = 'h-11 px-4 text-[15px]'

export function SignIn({ storedEmail }: { storedEmail: string | null }) {
  const queryClient = useQueryClient()
  const [email, setEmail] = useState(storedEmail ?? '')
  const [password, setPassword] = useState('')
  const [shown, setShown] = useState(false)

  const signIn = useMutation({
    mutationFn: async () => {
      const trimmed = email.trim()
      const reuseStored = !password && storedEmail && (!trimmed || trimmed === storedEmail)
      if (!trimmed && !storedEmail) throw new Error('Enter your BRAIN account email.')
      if (!password && !reuseStored) throw new Error('Enter your password.')
      const session = reuseStored ? await auth.login() : await auth.login(trimmed, password)
      if (!session.authenticated) {
        throw new ApiError(401, {
          code: session.verificationUrl ? 'verification_required' : 'not_authenticated',
          message: session.detail || 'That email and password were not accepted.',
          verificationUrl: session.verificationUrl ?? undefined,
        })
      }
      if (session.token && typeof window !== 'undefined') {
        localStorage.setItem('alpha_token', session.token)
      }
      return session
    },
    onSuccess: () => {
      setPassword('')
      useLive.setState({ verificationUrl: null })
      telemetry.reconnect()
      void queryClient.invalidateQueries()
    },
  })

  const needsVerification = signIn.error instanceof ApiError && Boolean(signIn.error.body.verificationUrl)

  return (
    <div className="flex min-h-svh flex-col items-center justify-center p-6">
      <div className="flex w-full max-w-[480px] flex-col gap-8">
        <div className="flex items-baseline gap-3 self-center">
          <span className="text-[30px] leading-none font-semibold text-primary" aria-hidden>
            α
          </span>
          <span className="text-[22px] leading-none font-medium tracking-[-0.4px]">Alpha Harness</span>
        </div>

        <form
          autoComplete="off"
          className="flex flex-col gap-5 rounded-xl border border-hairline bg-surface-1 p-10"
          onSubmit={(event: FormEvent) => {
            event.preventDefault()
            signIn.mutate()
          }}
        >
          <div className="mb-2 flex flex-col gap-2 text-center">
            <h1 className="headline">Sign in to BRAIN</h1>
            <p className="text-[15px] text-ink-subtle">Use your WorldQuant BRAIN account.</p>
          </div>

          <Field label="Email" className={FIELD}>
            <Input
              type="email"
              autoComplete="email"
              placeholder="you@example.com"
              required={!storedEmail}
              value={email}
              disabled={signIn.isPending}
              className={INPUT}
              onChange={(event) => {
                signIn.reset()
                setEmail(event.target.value)
              }}
            />
          </Field>
          <Field label="Password" className={FIELD}>
            <div className="relative">
              <Input
                id="brain-password"
                type={shown ? 'text' : 'password'}
                autoComplete="current-password"
                placeholder={storedEmail ? 'Leave blank to use the saved password' : 'Your BRAIN password'}
                value={password}
                disabled={signIn.isPending}
                className={`${INPUT} pr-18`}
                onChange={(event) => {
                  signIn.reset()
                  setPassword(event.target.value)
                }}
              />
              <button
                type="button"
                aria-pressed={shown}
                aria-controls="brain-password"
                onClick={() => setShown((s) => !s)}
                className="absolute inset-y-0 right-3 my-auto h-8 rounded-sm px-2 text-[13px] text-ink-subtle hover:text-ink"
              >
                {shown ? 'Hide' : 'Show'}
              </button>
            </div>
          </Field>

          {signIn.isError && (
            <ErrorNotice error={signIn.error} title={needsVerification ? 'BRAIN needs to verify your identity' : 'Sign-in failed'} />
          )}

          <Button type="submit" variant="primary" loading={signIn.isPending} className="mt-1 h-11 text-[15px]">
            {signIn.isPending ? 'Signing in…' : 'Sign in'}
          </Button>
        </form>
      </div>
    </div>
  )
}
