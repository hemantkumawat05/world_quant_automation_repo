/**
 * Navigation: the six areas, a live count of submittable Alphas, and the account menu.
 * Collapses to icons below `lg`.
 */

import { Fragment, useEffect } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Link, useRouterState } from '@tanstack/react-router'
import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { ExternalLinkIcon, LogOutIcon } from 'lucide-react'
import { toast } from 'sonner'
import { auth } from '@/api/core'
import { errorMessage, http } from '@/api/http'
import type { Today } from '@/api/types'
import { telemetry, useRefetchOn } from '@/lib/ws'
import { cx } from '@/ui/kit'
import { Menu } from '@/ui/overlay'
import { NAV } from './nav'

/** Areas a new consultant has to open once: Data (download fields) and AI (add a key). They flash until visited. */
const ONBOARDING: readonly string[] = ['data', 'ai']

const useVisited = create<{ visited: string[]; visit: (area: string) => void }>()(
  persist(
    (set) => ({
      visited: [],
      visit: (area) => set((s) => (s.visited.includes(area) ? s : { visited: [...s.visited, area] })),
    }),
    { name: 'alpha-harness-onboarding' },
  ),
)

export function Sidebar({ you, collapsed }: { you: Today['you']; collapsed: boolean }) {
  const queryClient = useQueryClient()
  const area = useRouterState({ select: (s) => s.location.pathname.split('/')[1] ?? '' })
  const visited = useVisited((s) => s.visited)
  const visit = useVisited((s) => s.visit)
  // Reaching the area any way counts (sidebar, ⌘K or a link), so the flash never outlives the visit.
  useEffect(() => {
    if (ONBOARDING.includes(area)) visit(area)
  }, [area, visit])
  const submittable = useQuery({
    queryKey: ['pool', 'submittable-count'],
    queryFn: () => http.get<{ total: number }>('/api/vault/submittable?limit=1'),
  })
  useRefetchOn('simulations', ['pool', 'submittable-count'], 5000)
  const total = submittable.data?.total ?? 0

  const signOut = useMutation({
    mutationFn: () => auth.logout(),
    onSuccess: () => {
      if (typeof window !== 'undefined') {
        localStorage.removeItem('alpha_token')
      }
      telemetry.reconnect()
      void queryClient.invalidateQueries()
    },
    onError: (error) => toast.error(errorMessage(error)),
  })

  const name = you.fullName ?? you.userId ?? 'Signed in'
  const initials = name
    .split(/\s+/)
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase()

  return (
    <aside className="flex h-full min-h-0 flex-col border-r border-hairline bg-canvas">
      <Link to="/dashboard" className={cx('flex h-12 shrink-0 items-center border-b border-hairline', collapsed ? 'justify-center' : 'px-4')} title="Alpha Harness">
        {/* Baseline, not centre: α is an x-height glyph, so centring its box drops it below the capitals. */}
        <span className="flex min-w-0 items-baseline gap-2.5">
          <span className="text-lg leading-none font-semibold text-primary" aria-hidden>
            α
          </span>
          {!collapsed && <span className="truncate text-[13px] leading-none font-medium tracking-[-0.2px]">Alpha Harness</span>}
        </span>
      </Link>

      <nav aria-label="Main" className="flex flex-1 flex-col gap-0.5 overflow-y-auto p-2">
        {NAV.map((item) => (
          <Fragment key={item.to}>
            <Link
              to={item.to}
              title={item.label}
              activeOptions={!collapsed && 'nested' in item ? { exact: true } : undefined}
              className={cx(
                'group flex h-8 items-center gap-2.5 rounded-md text-[13px] text-ink-subtle transition-colors hover:bg-surface-1 hover:text-ink data-[status=active]:bg-surface-2 data-[status=active]:text-ink',
                collapsed ? 'justify-center' : 'px-2',
                ONBOARDING.includes(item.area) && !visited.includes(item.area) && 'animate-attention text-warn motion-reduce:bg-warn/15',
              )}
            >
              <item.icon className="size-4 shrink-0 group-data-[status=active]:text-primary" aria-hidden />
              {!collapsed && <span className="flex-1 truncate">{item.label}</span>}
              {!collapsed && item.area === 'pool' && total > 0 && (
                <span className="num text-xs text-profit" title="Submittable Alphas">
                  {total}
                </span>
              )}
            </Link>
            {!collapsed && 'nested' in item && (
              // A guide line down from the parent's icon, so its children read as its pages.
              <div className="ml-4 flex flex-col gap-0.5 border-l border-hairline-strong pl-2">
                {item.tabs.map((sub) => (
                  <Link
                    key={sub.tab}
                    to={sub.to}
                    title={sub.label}
                    className="flex h-7 items-center rounded-md px-2 text-[13px] text-ink-subtle transition-colors hover:bg-surface-1 hover:text-ink data-[status=active]:bg-surface-2 data-[status=active]:text-ink"
                  >
                    <span className="truncate">{sub.label}</span>
                  </Link>
                ))}
              </div>
            )}
          </Fragment>
        ))}
      </nav>

      <div className="border-t border-hairline p-2">
        <Menu
          align="start"
          trigger={
            <button type="button" className={cx('flex h-10 w-full items-center gap-2.5 rounded-md text-left hover:bg-surface-1', collapsed ? 'justify-center' : 'px-1.5')}>
              <span className="num flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-3 text-[11px] text-ink-muted">{initials}</span>
              {!collapsed && (
                <span className="flex min-w-0 flex-col leading-tight">
                  <span className="truncate text-[13px]">{name}</span>
                  <span className="truncate text-xs text-ink-subtle">{you.email ?? you.userId}</span>
                </span>
              )}
            </button>
          }
          items={[
            {
              label: 'Open the BRAIN Platform',
              icon: <ExternalLinkIcon />,
              onClick: () => window.open('https://platform.worldquantbrain.com', '_blank', 'noopener,noreferrer'),
            },
            { label: 'Sign Out', icon: <LogOutIcon />, danger: true, disabled: signOut.isPending, onClick: () => signOut.mutate() },
          ]}
        />
      </div>
    </aside>
  )
}
