import { lazy, type ComponentType } from "react";

const RELOAD_KEY = "finpilot-chunk-reload";

/** `React.lazy` that survives deployments.
 *
 * After a rebuild the content-hashed chunk an already-open tab asks for no longer exists, and the
 * dynamic import rejects ("Failed to fetch dynamically imported module"). Reload once to pick up the
 * new index.html; if the import still fails, rethrow so the error boundary can show it.
 */
export function lazyWithRetry<T extends ComponentType<any>>(factory: () => Promise<{ default: T }>) {
  return lazy(async () => {
    try {
      const mod = await factory();
      try {
        sessionStorage.removeItem(RELOAD_KEY);
      } catch {
        /* storage unavailable */
      }
      return mod;
    } catch (err) {
      let reloaded = false;
      try {
        reloaded = sessionStorage.getItem(RELOAD_KEY) === "1";
        if (!reloaded) sessionStorage.setItem(RELOAD_KEY, "1");
      } catch {
        reloaded = true; // can't remember that we tried: don't risk a reload loop
      }
      if (!reloaded && isChunkError(err)) {
        window.location.reload();
        return new Promise<never>(() => {}); // the reload replaces this page
      }
      throw err;
    }
  });
}

export function isChunkError(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return /dynamically imported module|Importing a module script failed|Loading chunk|error loading dynamically/i.test(msg);
}
