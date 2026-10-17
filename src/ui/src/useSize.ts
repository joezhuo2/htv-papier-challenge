import { useLayoutEffect, useRef, useState } from 'react'

export function useSize<T extends HTMLElement>(fallback = { width: 640, height: 280 }) {
  const ref = useRef<T>(null)
  const [size, setSize] = useState(fallback)
  useLayoutEffect(() => {
    const el = ref.current!
    const read = () => {
      const r = el.getBoundingClientRect()
      if (r.width && r.height) setSize({ width: r.width, height: r.height })
    }
    read()
    const ro = new ResizeObserver(read)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  return [ref, size] as const
}
