import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import { App } from './App'
import { parseSource } from './source'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App source={parseSource(window.location.search)} />
  </StrictMode>,
)
