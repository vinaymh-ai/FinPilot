const TOKEN_KEY = "finpilot-token";

export class ApiError extends Error {
  status: number;
  detail: unknown;
  constructor(status: number, message: string, detail?: unknown) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

function authHeaders(): Record<string, string> {
  try {
    const t = localStorage.getItem(TOKEN_KEY);
    return t ? { Authorization: `Bearer ${t}` } : {};
  } catch {
    return {};
  }
}

/** Fired when the session is missing or expired, so the app can show the sign-in screen. */
export const UNAUTHENTICATED_EVENT = "finpilot:unauthenticated";

async function request<T>(method: string, path: string, body?: unknown, isForm = false): Promise<T> {
  // X-Requested-With is the server's CSRF check for cookie-authenticated writes.
  const headers: Record<string, string> = { "X-Requested-With": "FinPilot", ...authHeaders() };
  if (body !== undefined && !isForm) headers["Content-Type"] = "application/json";
  const res = await fetch(`/api${path}`, {
    method,
    headers,
    credentials: "same-origin",
    body: body === undefined ? undefined : isForm ? (body as FormData) : JSON.stringify(body),
  });
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  let data: any = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!res.ok) {
    const detail = data?.detail ?? data;
    let msg = typeof detail === "string" ? detail : detail?.message ?? `Request failed (${res.status})`;
    const requestId = detail?.request_id ?? res.headers.get("x-request-id");
    if (res.status >= 500 && requestId) msg = `${msg} (request id ${requestId})`;
    if (res.status === 401 && detail?.code === "not_authenticated" && !path.startsWith("/auth/")) {
      window.dispatchEvent(new Event(UNAUTHENTICATED_EVENT));
    }
    throw new ApiError(res.status, msg, detail);
  }
  return data as T;
}

export const api = {
  get: <T>(p: string) => request<T>("GET", p),
  post: <T>(p: string, b?: unknown) => request<T>("POST", p, b ?? {}),
  put: <T>(p: string, b: unknown) => request<T>("PUT", p, b),
  patch: <T>(p: string, b: unknown) => request<T>("PATCH", p, b),
  del: <T>(p: string) => request<T>("DELETE", p),
  upload: <T>(p: string, form: FormData) => request<T>("POST", p, form, true),
  setToken(token: string) {
    try {
      if (token) localStorage.setItem(TOKEN_KEY, token);
      else localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* storage unavailable */
    }
  },
};
